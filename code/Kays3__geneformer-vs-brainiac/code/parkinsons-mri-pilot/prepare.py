"""Rigidly register T1 scans and apply a common MNI template mask."""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import SimpleITK as sitk
import nilearn

ROOT = Path(nilearn.__file__).parent / "datasets/data"
TEMPLATE = next(ROOT.rglob("mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz"))


def prepare(path):
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(2)
    fixed = sitk.ReadImage(str(TEMPLATE), sitk.sitkFloat32)
    fixed = sitk.Resample(fixed, [round(s / 2) for s in fixed.GetSize()],
                          sitk.Transform(), sitk.sitkLinear, fixed.GetOrigin(),
                          [v * 2 for v in fixed.GetSpacing()], fixed.GetDirection())
    mask = sitk.Cast(fixed > 0, sitk.sitkUInt8)
    moving = sitk.ReadImage(str(path), sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(32)
    reg.SetMetricFixedMask(mask)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.20, 197)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(2.0, 0.01, 100)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SetInitialTransform(initial)
    transform = reg.Execute(fixed, moving)
    aligned = sitk.Mask(sitk.Resample(moving, fixed, transform, sitk.sitkLinear), mask)
    sitk.WriteImage(aligned, str(Path("aligned") / path.name))


if __name__ == "__main__":
    Path("aligned").mkdir(exist_ok=True)
    with ProcessPoolExecutor(min(4, os.cpu_count() or 1)) as pool:
        list(pool.map(prepare, sorted(Path("data").glob("*nii.gz"))))
