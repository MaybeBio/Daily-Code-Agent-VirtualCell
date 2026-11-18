<script lang="ts">
  import { onMount } from 'svelte';
  import { goto } from '$app/navigation';
  import PageHeader from '$lib/components/layout/PageHeader.svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import ToolResultView from '$lib/components/ui/ToolResultView.svelte';
  import FilePickerPopup from '$lib/components/ui/FilePickerPopup.svelte';
  import TerminalOutput from '$lib/components/ui/TerminalOutput.svelte';
  import RunRecord from '$lib/components/ui/RunRecord.svelte';
  import DepCheck, { type DepStatus } from '$lib/components/ui/DepCheck.svelte';
  import SnpEffSuiteManager from '$lib/components/tools/SnpEffSuiteManager.svelte';
  import { DEP_REQUIREMENTS } from '$lib/data/dep-requirements';
  import { liatir } from '$lib/api';
  import { beginDirectNativeToolRun } from '$lib/execution/direct-native-tool';
  import { ensureRunOutputDir } from '$lib/execution/run-storage';
  import { dataFiles } from '$lib/stores/dataFiles.svelte';
  import { analysisRuns } from '$lib/stores/analysisRuns.svelte';
  import { executionRuns } from '$lib/stores/executionRuns.svelte';
  import { confirm } from '$lib/stores/confirm.svelte';
  import { snpEffStore } from '$lib/stores/snpeff.svelte';
  import { settingsStore } from '$lib/stores/settings.svelte';
  import { fmtDuration, sanitizeLocalPathsForDisplay } from '$lib/utils';
  import { runNativeTool } from '$lib/utils/native-tool';
  import { notify } from '$lib/utils/notify';
  import { buildSnpEffOutput, parseSnpEffStats } from '$lib/tools/variants/snpeff';
  import type { ToolOutput } from '$lib/types/tool-output';
  import type { RunOutputFile } from '$lib/stores/analysisRuns.svelte';

  const TOOL_ID = 'snpeff';
  let depStatus = $state<DepStatus>('checking');
  let selectedGenome = $state('GRCh38.115');
  let useExternal = $state(false);
  let externalGenome = $state('GRCh38.115');
  let externalGenomePresent = $state(false);
  let externalChecking = $state(false);
  let filePath = $state('');
  let running = $state(false);
  let startedAt = $state<number | null>(null);
  let now = $state(Date.now());
  let logLines = $state<string[]>([]);
  let activeExecutionRunId = $state<string | null>(null);
  let selectedRunId = $state<string | null>(null);
  let loadedOutput = $state<ToolOutput | null>(null);
  let loadingOutput = $state(false);

  const runs = $derived(analysisRuns.byTool(TOOL_ID));
  const vcfFiles = $derived(dataFiles.byExt('vcf', 'vcf.gz'));
  const selectedRun = $derived(runs.find((run) => run.id === selectedRunId) ?? null);
  const selectedRunOutputFiles = $derived(selectedRun?.outputFiles ?? []);
  const selectedCatalogEntry = $derived(snpEffStore.catalog?.databases.find((entry) =>
    entry.id === selectedGenome && entry.suiteVersion === snpEffStore.active?.version
  ) ?? null);
  const managedDatabaseReady = $derived(Boolean(
    selectedCatalogEntry && snpEffStore.installedDatabaseFor(selectedCatalogEntry)
  ));
  const externalReady = $derived(Boolean(
    snpEffStore.config.externalJarPath && externalGenome.trim() && externalGenomePresent
  ));
  const runtimeReady = $derived(useExternal ? externalReady : Boolean(snpEffStore.active && managedDatabaseReady));
  const displayError = $derived<string | null>(
    selectedRun?.status === 'error' ? (selectedRun.error ?? 'Unknown error') : null,
  );

  $effect(() => {
    if (!running) return;
    const id = setInterval(() => now = Date.now(), 1000);
    return () => clearInterval(id);
  });

  $effect(() => {
    const id = selectedRunId;
    if (!id) { loadedOutput = null; return; }
    loadingOutput = true;
    analysisRuns.loadOutput(id).then((output) => {
      loadedOutput = output;
      loadingOutput = false;
    });
  });

  $effect(() => {
    const genome = externalGenome.trim();
    const jar = snpEffStore.config.externalJarPath;
    const dataDir = snpEffStore.config.externalDataDir;
    if (!useExternal || !jar || !genome || !dataDir) {
      externalGenomePresent = false;
      externalChecking = false;
      return;
    }
    let current = true;
    externalChecking = true;
    snpEffStore.checkExternalGenomePresent(genome).then((present) => {
      if (!current) return;
      externalGenomePresent = present;
      externalChecking = false;
    });
    return () => { current = false; };
  });

  onMount(async () => {
    dataFiles.init();
    analysisRuns.init();
    await Promise.all([snpEffStore.init(), settingsStore.init()]);
  });

  function basename(path: string) {
    return path.split(/[\\/]/).pop() ?? path;
  }

  async function browseExternalJar() {
    const api = liatir();
    if (!api) return;
    try {
      const result = await api.invoke('lia_file_open', {
        multi: false,
        allowedExtensions: ['jar'],
      }) as { paths: string[] };
      const path = result.paths[0];
      if (path) await snpEffStore.setExternalJarPath(path);
    } catch { /* picker cancelled */ }
  }

  async function runAnnotation() {
    const genome = useExternal ? externalGenome.trim() : selectedGenome;
    if (!filePath || !genome || running) return;

    let runtime;
    try {
      runtime = await snpEffStore.captureSnpEffRuntime(genome, useExternal);
    } catch (error) {
      await notify('SnpEff is not ready', String(error));
      return;
    }

    const api = liatir();
    if (!api) return;
    running = true;
    selectedRunId = null;
    startedAt = Date.now();
    const runId = crypto.randomUUID();
    const t0 = startedAt;
    const fileName = basename(filePath);
    const fileSize = dataFiles.files.find((file) => file.path === filePath)?.size;
    const inputSizes = fileSize == null ? undefined : [fileSize];
    const params = {
      genome,
      heap: snpEffStore.jvmHeap,
      snpeffSuiteVersion: runtime.suiteVersion,
      snpeffSuiteSha256: runtime.suiteArchiveSha256,
      snpeffSuiteSource: runtime.source,
      snpeffDatabaseId: runtime.database!.id,
      snpeffDatabaseSeries: runtime.database!.databaseSeries,
      snpeffDatabaseSha256: runtime.database!.archiveSha256,
      snpeffDatabaseSource: runtime.database!.source,
    };
    const execution = await beginDirectNativeToolRun({
      runId,
      toolId: TOOL_ID,
      label: fileName,
      inputs: [filePath],
      params,
      startedAt: t0,
    }).catch(async (error) => {
      await notify('SnpEff failed', String(error));
      return null;
    });
    if (!execution) {
      running = false;
      startedAt = null;
      return;
    }
    activeExecutionRunId = runId;

    try {
      const outDir = await ensureRunOutputDir(runId);
      const outPath = `${outDir}/annotated.vcf`;
      const statsBase = outPath.replace(/\.vcf$/, '');
      const statsHtml = `${statsBase}-summary.html`;
      const statsGenes = `${statsBase}-summary.genes.txt`;
      const java = settingsStore.javaPath || 'java';
      logLines = [
        `$ java -Xmx${snpEffStore.jvmHeap} -jar snpEff.jar ann ${genome} ${fileName}`,
        `→ ${runtime.source === 'managed' ? 'Verified Liatir suite' : 'External unverified suite'}`,
      ];
      const result = await runNativeTool(
        java,
        [
          `-Xmx${snpEffStore.jvmHeap}`,
          '-jar', runtime.snpEffJar,
          'ann',
          '-dataDir', runtime.dataDir,
          '-noLog',
          '-stats', statsHtml,
          genome,
          filePath,
        ],
        undefined,
        (line) => { if (line.trim() && logLines.length < 500) logLines.push(line); },
        execution.nativeOptions({
          stdoutPath: outPath,
          metadata: {
            snpeffSuiteVersion: runtime.suiteVersion,
            snpeffSuiteSha256: runtime.suiteArchiveSha256,
            snpeffSuiteSource: runtime.source,
            snpeffDatabaseId: runtime.database!.id,
            snpeffDatabaseSha256: runtime.database!.archiveSha256,
          },
        }),
      );
      if (!result.ok) throw new Error(result.stderr || `SnpEff exited with code ${result.exitCode}`);

      const outputFiles: RunOutputFile[] = [{
        label: 'Annotated VCF',
        path: outPath,
        ext: 'vcf',
        size: await api.invoke('lia_file_size', { path: outPath }) as number,
      }];
      const sideEffects: RunOutputFile[] = [];
      for (const [label, path, ext] of [
        ['Summary (HTML)', statsHtml, 'html'],
        ['Gene stats', statsGenes, 'txt'],
      ] as const) {
        try {
          sideEffects.push({ label, path, ext, size: await api.invoke('lia_file_size', { path }) as number });
        } catch { /* optional SnpEff diagnostic */ }
      }
      const summary = parseSnpEffStats(result.stderr);
      const endedAt = Date.now();
      logLines.push(`✓ Annotation complete in ${fmtDuration(t0, endedAt)}`);
      await snpEffStore.touchGenome(genome);
      await execution.finalize('done', {
        id: runId,
        tool: TOOL_ID,
        label: fileName,
        inputs: [filePath],
        inputSizes,
        params,
        startedAt: t0,
        endedAt,
        durationMs: endedAt - t0,
        output: buildSnpEffOutput(summary, fileName),
        outputFiles,
        sideEffects,
        error: null,
        log: [...logLines],
      });
      await notify('SnpEff complete', `${fileName} finished in ${fmtDuration(t0, endedAt)}`, endedAt - t0);
    } catch (error) {
      const endedAt = Date.now();
      const cancelled = execution.isCancelled(error);
      const message = cancelled ? 'SnpEff run was cancelled.' : String(error);
      logLines.push(cancelled ? `■ ${message}` : `✗ Error: ${message}`);
      await execution.finalize(cancelled ? 'cancelled' : 'error', {
        id: runId,
        tool: TOOL_ID,
        label: fileName,
        inputs: [filePath],
        inputSizes,
        params,
        startedAt: t0,
        endedAt,
        durationMs: endedAt - t0,
        output: null,
        outputFiles: [],
        sideEffects: [],
        error: message,
        log: [...logLines],
      });
      await notify(cancelled ? 'SnpEff cancelled' : 'SnpEff failed', message);
    } finally {
      running = false;
      startedAt = null;
      activeExecutionRunId = null;
      selectedRunId = runId;
    }
  }

  async function deleteRun(id: string, label: string) {
    const accepted = await confirm({
      title: 'Delete run',
      message: `Delete the run for "${label}"?`,
      confirmLabel: 'Delete',
    });
    if (!accepted) return;
    if (selectedRunId === id) selectedRunId = runs.find((run) => run.id !== id)?.id ?? null;
    await analysisRuns.remove(id);
  }

  function fmtDate(ms: number) {
    return new Date(ms).toLocaleDateString([], {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
  }
</script>

<div class="flex h-full overflow-hidden">
  <div class="w-52 shrink-0 border-r border-border bg-surface flex flex-col">
    <div class="flex items-center justify-between px-3 py-3 border-b border-border">
      <span class="text-xs font-medium text-text-secondary">Run history</span>
      {#if runs.length > 0}<span class="text-[10px] text-text-subtle">{runs.length}</span>{/if}
    </div>
    <div class="flex-1 overflow-y-auto py-1">
      {#if runs.length === 0}
        <p class="text-xs text-text-subtle text-center py-8 px-3 leading-relaxed">No runs yet.<br/>Results will appear here.</p>
      {:else}
        {#each runs as run (run.id)}
          <div class="group relative flex items-start transition-colors {selectedRunId === run.id ? 'bg-brand/8' : 'hover:bg-surface-2'}">
            <button onclick={() => selectedRunId = run.id} class="flex-1 text-left px-3 py-2.5 min-w-0">
              <div class="flex items-center gap-1.5 mb-0.5">
                <span class="h-1.5 w-1.5 rounded-full shrink-0 {run.status === 'done' ? 'bg-emerald-500' : 'bg-red-500'}"></span>
                <p class="text-xs font-medium truncate {selectedRunId === run.id ? 'text-brand' : 'text-text-secondary'}">{run.label}</p>
              </div>
              <p class="text-[10px] text-text-subtle pl-3">{fmtDate(run.startedAt)} · {fmtDuration(run.startedAt, run.endedAt)}</p>
            </button>
            <button onclick={() => deleteRun(run.id, run.label)} aria-label="Delete run" class="opacity-0 group-hover:opacity-100 p-1.5 mt-2 mr-1.5 text-text-subtle hover:text-red-500 rounded">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>
        {/each}
      {/if}
    </div>
  </div>

  <div class="flex-1 flex flex-col overflow-hidden">
    <PageHeader title="SnpEff" description="Annotate variants with their predicted biological effects">
      {#snippet actions()}
        <Button variant="ghost" size="sm" onclick={() => goto('/tools')}>← Back</Button>
      {/snippet}
    </PageHeader>

    <div class="flex-1 overflow-y-auto p-6 space-y-5">
      <DepCheck req={DEP_REQUIREMENTS.java} onStatusChange={(status) => depStatus = status} />

      <SnpEffSuiteManager
        showDatabases
        selectedDatabaseId={selectedGenome}
        onselect={(id) => selectedGenome = id}
      />

      {#if depStatus === 'ok'}
        <details class="rounded-xl border border-border bg-surface p-4">
          <summary class="cursor-pointer text-xs font-medium text-text-secondary">Advanced: use an existing external installation</summary>
          <div class="mt-4 space-y-3">
            <p class="text-[11px] text-amber-700 leading-relaxed">
              External files are kept for compatibility, but Liatir cannot verify their version or origin.
            </p>
            <label class="flex items-center gap-2 text-xs text-text-secondary">
              <input type="checkbox" bind:checked={useExternal} /> Use external SnpEff for this run
            </label>
            <div class="flex items-center gap-2">
              <Button variant="secondary" size="sm" onclick={browseExternalJar}>Select snpEff.jar</Button>
              <span class="min-w-0 truncate text-[11px] font-mono text-text-subtle">
                {snpEffStore.config.externalJarPath ?? 'No external JAR selected'}
              </span>
            </div>
            <label class="block text-[11px] text-text-muted">
              External database folder
              <input
                value={snpEffStore.config.externalDataDir}
                onchange={(event) => snpEffStore.setExternalDataDir(event.currentTarget.value)}
                class="mt-1 w-full rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-brand/30"
              />
            </label>
            <label class="block text-[11px] text-text-muted">
              External genome ID
              <input bind:value={externalGenome} class="mt-1 w-full rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-brand/30" />
            </label>
            {#if useExternal}
              <p class="text-[11px] {externalGenomePresent ? 'text-emerald-700' : 'text-amber-700'}">
                {externalChecking ? 'Checking external database…' : externalGenomePresent ? 'External database found' : 'External database not found'}
              </p>
            {/if}
          </div>
        </details>

        <Card class="p-5 space-y-4">
          <div>
            <h2 class="text-sm font-semibold text-text">Annotate variants</h2>
            <p class="mt-1 text-xs text-text-secondary">
              The selected database identity and exact suite checksum are saved with the result.
            </p>
          </div>

          {#if !runtimeReady}
            <div class="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
              {useExternal
                ? 'Select an external JAR and a database that is present in its data folder.'
                : 'Install the suite and the selected verified genome database above.'}
            </div>
          {/if}

          <div class="flex items-center gap-3">
            <span class="text-[11px] text-text-muted shrink-0">Java memory</span>
            <div class="flex flex-wrap gap-1.5">
              {#each ['4g', '6g', '8g', '12g', '16g'] as heap}
                <button
                  onclick={() => snpEffStore.setJvmHeap(heap)}
                  class="px-2.5 py-1 rounded-md border text-xs {snpEffStore.jvmHeap === heap ? 'bg-brand text-white border-brand' : 'bg-surface border-border text-text-secondary'}"
                >{heap.replace('g', ' GB')}</button>
              {/each}
            </div>
          </div>

          <FilePickerPopup files={vcfFiles} value={filePath} label="VCF file" emptyText="No VCF files in Data yet." disabled={running} onchange={(path) => filePath = path} />
          <div class="flex items-center gap-3">
            <Button variant="primary" testId="direct-native-run" disabled={!filePath || !runtimeReady || running} loading={running} onclick={runAnnotation}>Run annotation</Button>
            {#if running && activeExecutionRunId}
              <Button variant="secondary" testId="direct-native-cancel" onclick={() => executionRuns.cancel(activeExecutionRunId!)}>Cancel</Button>
            {/if}
            {#if running && startedAt}<span class="text-xs text-text-subtle">Elapsed: {fmtDuration(startedAt, now)}</span>{/if}
          </div>
          <TerminalOutput lines={logLines} {running} />
        </Card>

        {#if displayError}
          <div class="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 font-mono">{sanitizeLocalPathsForDisplay(displayError, 2)}</div>
          <RunRecord runId={selectedRunId} />
        {:else if loadingOutput}
          <div class="flex justify-center py-12"><span class="text-xs text-text-subtle">Loading result…</span></div>
        {:else if loadedOutput}
          <div>
            <div class="flex items-center justify-between mb-3">
              <h2 class="text-xs font-medium text-text-secondary">{selectedRun?.label ?? 'Result'}</h2>
              {#if selectedRun}<span class="text-xs text-text-subtle">{fmtDate(selectedRun.startedAt)} · {fmtDuration(selectedRun.startedAt, selectedRun.endedAt)}</span>{/if}
            </div>
            <ToolResultView output={loadedOutput} outputFiles={selectedRunOutputFiles} />
            <RunRecord runId={selectedRunId} />
          </div>
        {/if}
      {/if}
    </div>
  </div>
</div>
