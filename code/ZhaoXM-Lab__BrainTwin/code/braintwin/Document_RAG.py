import os
import re
import csv
import json
import logging
import warnings
from typing_extensions import TypedDict, List, Annotated

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define state and query types
class State(TypedDict):
    question: str
    query: dict
    context: List
    answer: str
    sources: List[str]


class AcademicQuery(TypedDict):
    query: Annotated[str, "Search query"]
    year_range: Annotated[str, "Publication year range, for example 2020-2023 or any"]
    specific_paper: Annotated[str, "Paper title, or none"]


def load_vector_store(
    model_path=None,
    store_path=None):
    """Load the existing vector store."""
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if model_path is None:
        model_path = os.path.join(root, "data", "embedding", "multilingual-e5-large-instruct")
    if store_path is None:
        store_path = os.path.join(root, "data", "literature_index")
    print(f"load vector storage from {store_path}...")
    try:
        print("Initializing Embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name = model_path,
            model_kwargs = {'device': 'cuda'}
        )
        print("Initialization success.")
        if not os.path.exists(store_path):
            print(f"Warning: Storage path does not exist: {store_path}")
            return None
        print("Loading vector storage...")
        vector_store = FAISS.load_local(
            store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("Loading success.")
        print(f"There are {vector_store.index.ntotal} document indexes.")
        return vector_store
    except Exception as e:
        print(f"Error in loading: {str(e)}")
        return None


def get_document_group_key(doc):
    """Create a grouping key for documents based on source and page"""
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    return f"{source}_{page}"


def group_similar_chunks(docs, max_docs=20):
    """Group chunks from the same document page and limit total docs"""
    if not docs:
        return []
    doc_groups = {}
    for doc in docs:
        group_key = get_document_group_key(doc)
        if group_key not in doc_groups:
            doc_groups[group_key] = []
        doc_groups[group_key].append(doc)
    merged_docs = []
    for group_key, group_docs in doc_groups.items():
        if len(group_docs) == 1:
            merged_docs.append(group_docs[0])
        else:
            group_docs.sort(key=lambda d: d.metadata.get("start_index", 0))
            combined_content = "\n".join(d.page_content for d in group_docs)
            merged_doc = group_docs[0]
            merged_doc.page_content = combined_content
            merged_docs.append(merged_doc)
    if len(merged_docs) > max_docs:
        merged_docs = merged_docs[:max_docs]
    return merged_docs


ANATOMY_TERMS = (
    "pars opercularis", "pars triangularis", "pars orbitalis",
    "rostral middle frontal", "caudal middle frontal", "superior frontal",
    "fusiform", "superior temporal", "middle temporal", "inferior temporal",
    "temporal pole", "entorhinal", "parahippocamp", "inferior parietal",
    "superior parietal", "precuneus", "posterior cingulate", "anterior cingulate",
    "insula", "hippocamp", "thalam", "amygdala", "caudate", "putamen",
    "pallidum", "accumbens", "lateral ventricle",
)
METRIC_WORDS = (
    "thickness", "volume", "atrophy", "thinner", "surface area",
    "gray matter", "grey matter",
)


def is_region_question(question: str) -> bool:
    text = question.lower()
    return "structural mri" in text and "region" in text


def term_in(text: str, term: str) -> bool:
    if len(term) <= 4:
        return re.search(rf"\b{re.escape(term)}\b", text) is not None
    return term in text


def disease_terms(question: str):
    text = question.lower()
    terms = []
    if "schizophrenia" in text:
        terms.extend(["schizophrenia", "scz"])
    if "bipolar" in text:
        terms.extend(["bipolar", "bd"])
    if "adhd" in text or "attention-deficit" in text or "attention deficit" in text:
        terms.extend(["adhd", "attention-deficit", "attention deficit"])
    if "alzheimer" in text:
        terms.append("alzheimer")
    if "depress" in text:
        terms.extend(["depress", "mdd"])
    if "anxiety" in text or "panic" in text:
        terms.extend(["anxiety", "panic"])
    if "epilep" in text:
        terms.append("epilep")
    if "ptsd" in text or "posttraumatic" in text or "post-traumatic" in text:
        terms.extend(["ptsd", "posttraumatic", "post-traumatic"])
    if "parkinson" in text:
        terms.append("parkinson")
    if "stroke" in text:
        terms.append("stroke")
    return terms


def sentence_around(text: str, start: int, end: int) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("? ", 0, start))
    right = text.find(". ", end)
    begin = 0 if left < 0 else left + 2
    stop = len(text) if right < 0 else right + 1
    return " ".join(text[begin:stop].split())


def is_reference_list(text: str) -> bool:
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", text))
    citations = len(re.findall(r"et al", text, re.I))
    words = max(len(text.split()), 1)
    return years >= 6 and (years + citations) / words > 0.025


def collect_region_passages(vector_store, question: str, limit: int = 8):
    """Keep sentences in which the disease, a named structure, and a metric are close."""
    terms = disease_terms(question)
    docstore = getattr(vector_store, "docstore", None)
    documents = getattr(docstore, "_dict", None) or {}
    if not terms or not documents:
        return []
    best_by_page = {}
    for doc in documents.values():
        text = doc.page_content or ""
        if is_reference_list(text):
            continue
        lowered = text.lower()
        meta = doc.metadata or {}
        title = (meta.get("title") or "").lower()
        paper_about_disease = any(term_in(title, term) for term in terms)
        if not paper_about_disease and not any(term_in(lowered, term) for term in terms):
            continue
        if not any(word in lowered for word in METRIC_WORDS):
            continue
        for region in ANATOMY_TERMS:
            start = lowered.find(region)
            if start < 0:
                continue
            window = sentence_around(text, start, start + len(region))
            window_l = window.lower()
            if not paper_about_disease and not any(term_in(window_l, term) for term in terms):
                continue
            if not any(word in window_l for word in METRIC_WORDS) and not paper_about_disease:
                continue
            present = sum(name in window_l for name in ANATOMY_TERMS)
            meta = doc.metadata or {}
            page_key = (meta.get("source", ""), meta.get("page", ""), window[:80])
            current = best_by_page.get(page_key)
            if current is None or present > current[0]:
                best_by_page[page_key] = (present, window, doc, region)
    def covered(window: str) -> set:
        text_l = window.lower()
        return {name for name in ANATOMY_TERMS if name in text_l}

    def about_disease(doc) -> bool:
        title = ((doc.metadata or {}).get("title") or "").lower()
        return any(term_in(title, term) for term in terms)

    def passage_rank(item) -> tuple:
        score, window, doc, region = item
        lowered = window.lower()
        correlation = "correlat" in lowered or "partial r" in lowered
        effect = any(word in lowered for word in ("thinner", "smaller", "larger", "effect size", "cohen"))
        cortical = any(word in lowered for word in ("thickness", "thinner", "gyrus", "gyri"))
        return (about_disease(doc), effect, cortical, not correlation, score)

    ranked = sorted(best_by_page.values(), key=passage_rank, reverse=True)
    passages = []
    seen_text = set()
    seen_regions = set()
    for score, window, doc, region in ranked:
        lowered = window.lower()
        if "correlat" in lowered or "partial r" in lowered:
            continue
        if window in seen_text:
            continue
        found = covered(window)
        if seen_regions and not (found - seen_regions):
            continue
        seen_text.add(window)
        seen_regions.update(found)
        meta = dict(doc.metadata or {})
        meta["matched_region"] = region
        passages.append(Passage(window, meta))
        if len(passages) >= limit:
            break
    passages.sort(
        key=lambda passage: (
            "largest" in passage.page_content.lower() and ("thinner" in passage.page_content.lower() or "thickness" in passage.page_content.lower()),
            "thinner" in passage.page_content.lower() or "thickness" in passage.page_content.lower(),
            "smaller" in passage.page_content.lower() or "larger" in passage.page_content.lower(),
        ),
        reverse=True,
    )
    if not passages:
        passages.append(Passage(
            "No structural MRI sentence in the index linked a named gyrus or nucleus to this disease.",
            {"title": "none", "source": "", "page": ""},
        ))
    return passages


class Passage:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def extract_search_query(llm, question: str, verbose: bool = False) -> dict:
    """Extract retrieval keywords via plain LLM invoke + JSON parse (no structured output)."""
    default = {
        "query": question.strip(),
        "year_range": "any",
        "specific_paper": "none",
    }
    prompt_text = (
        "Extract search keywords from the following question and its options. "
        "Your response MUST be ONLY a valid JSON object with a 'query' key.\n"
        "IMPORTANT: If any keywords contain double quotes, you MUST escape them with a backslash (e.g., \\\"text\\\").\n"
        'Example format: {"query": "keywords from the question"}\n\n'
        f"Question: {question}"
    )
    try:
        response = llm.invoke(prompt_text)
        text = response.content if hasattr(response, "content") else str(response)
        text = text.strip()

        if "```" in text:
            block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if block:
                text = block.group(1).strip()
        if not text.startswith("{"):
            block = re.search(r"\{[\s\S]*\}", text)
            if block:
                text = block.group(0)

        parsed = json.loads(text)
        query_text = parsed.get("query") or question
        result = {
            "query": str(query_text).strip(),
            "year_range": parsed.get("year_range", "any"),
            "specific_paper": parsed.get("specific_paper", "none"),
        }
        if verbose:
            print(f"Parsed query: {result}")
        return result
    except Exception as exc:
        if verbose:
            print(f"Query parse failed, using original question: {exc}")
        return default


def build_rag_application(llm, vector_store):
    """Build RAG application using custom API"""
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a neuroscientist with extensive experience in medical imaging and clinical diagnosis.\n"
         "Use the retrieved context when it is relevant. If no useful context is provided, answer from your own knowledge.\n"
         "If the user question is a multiple-choice question with options A/B/C/D, reply with ONLY the best option letter and a one-sentence justification.\n"
         "If the user question is open-ended, reply with a clear, concise paragraph (3-6 sentences) explaining the answer in plain language.\n"
         "Name each brain region at most once. Stop when the list is complete. Do not repeat a region or a sentence.\n"
         "Do NOT invent new multiple-choice options unless they already exist in the user's question."),
        ("human", "Context (if available):\n{context}\n\nQuestion: {question}")
    ])
    region_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are listing structural MRI regions for one disease.\n"
         "First list every gyrus or subcortical nucleus in the context that the passage links to this disease.\n"
         "Then add a region only when a structural MRI study reports it for this disease.\n"
         "Name gyri and nuclei separately. Do not replace them with a lobe, prefrontal cortex, or basal ganglia.\n"
         "Include both cortical regions and subcortical nuclei named in the context.\n"
         "If a sentence lists several subdivisions of one lobe, name each subdivision separately.\n"
         "A parenthesis such as temporal (inferior, middle, and superior) is three gyri, not one region.\n"
         "Do not list the left and right copies of one structure as two regions.\n"
         "Do not split a nucleus into unnamed subnuclei.\n"
         "Say cortical thickness or volume, and the direction when it is known.\n"
         "If the context says no sentence was found, list the gyri and nuclei reported by structural MRI studies of this disease.\n"
         "Write at most 12 regions, one mention each, then stop. Do not repeat a region."),
        ("human", "Context (if available):\n{context}\n\nQuestion: {question}")
    ])

    def analyze_query(state: State, verbose=False):
        """Generate a search query without with_structured_output."""
        if verbose:
            print(f"Analyzing query: {state['question']}")
        query = extract_search_query(llm, state["question"], verbose=verbose)
        if verbose:
            print(f"Structured query: {query}")
        return {"query": query}

    def retrieve(state: State, verbose=False):
        """Retrieves documents from the vector store."""
        query_text = state["query"]["query"]
        if verbose:
            print(f"Retrieving for query: {query_text}")
        if is_region_question(state["question"]):
            region_docs = collect_region_passages(vector_store, state["question"])
            if region_docs:
                if verbose:
                    print(f"Using {len(region_docs)} disease-region sentences. Reference-list pages were skipped.")
                sources = []
                for doc in region_docs:
                    title = doc.metadata.get("title", "unknown")
                    source = doc.metadata.get("source", "")
                    page = doc.metadata.get("page", "")
                    sources.append(f"{title} - {source}, page {page}")
                return {"context": region_docs, "sources": sources}
        try:
            retrieved_docs = vector_store.similarity_search(query_text, k=15)
            processed_docs = group_similar_chunks(retrieved_docs, max_docs=10)
            if len(processed_docs) < 3:
                print("Few hits. Trying another query.")
                keywords = " ".join([word for word in query_text.split() if len(word) > 3])
                if keywords and keywords != query_text:
                    additional_docs = vector_store.similarity_search(keywords, k=10)
                    existing_keys = set(get_document_group_key(doc) for doc in processed_docs)
                    for doc in additional_docs:
                        if get_document_group_key(doc) not in existing_keys:
                            processed_docs.append(doc)
                            if len(processed_docs) >= 10: break
            if verbose:
                print(f"Using {len(processed_docs)} documents.")
        except Exception as e:
            if verbose:
                print(f"Retrieval failed: {str(e)}")
            processed_docs = []
        sources = []
        for doc in processed_docs:
            title = doc.metadata.get("title", "unknown")
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            year = doc.metadata.get("year", "unknown year")
            if year == "unknown year" and "source" in doc.metadata:
                filename = os.path.basename(doc.metadata["source"])
                year_match = re.search(r'(\d{4})', filename)
                if year_match: year = year_match.group(1)
            source_str = f"{title} ({year}) - {source}, page {page}" if page else f"{title} ({year}) - {source}"
            sources.append(source_str)
        return {"context": processed_docs, "sources": sources}

    def generate(state: State, verbose=False):
        """Generates an answer using the LLM."""
        context_docs = state['context']
        if context_docs:
            if verbose:
                print(f"Generating an answer from {len(context_docs)} documents.")
            docs_content = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
            max_chars = 4000
            if len(docs_content) > max_chars:
                if verbose:
                    print(f"Context was truncated to {max_chars} characters.")
                docs_content = docs_content[:max_chars] + "...[truncated]"
        else:
            if verbose:
                print("No relevant context was found.")
            docs_content = "No specific context found."
        chosen = region_prompt if is_region_question(state["question"]) else prompt
        messages = chosen.invoke({"question": state["question"], "context": docs_content})
        response = llm.invoke(messages, max_tokens=500, frequency_penalty=0.6)
        return {"answer": response.content}

    graph_builder = StateGraph(State)
    graph_builder.add_node("analyze_query", analyze_query)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "analyze_query")
    graph_builder.add_edge("analyze_query", "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    return graph_builder.compile()