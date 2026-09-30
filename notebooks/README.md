# Research notebook

[Back to project overview](../README.md)

[Deepfake AI Detection and Source Attribution](Deepfake_AI_Detection_and_Source_Attribution.ipynb) contains the original research workflow: dataset discovery, cepstral feature extraction, dual-head LCNN training, classification reports, confusion matrices, and Grad-CAM exploration.

The notebook is preserved unchanged, including its recorded outputs. It was written for Kaggle and references `/kaggle/input` datasets and `/kaggle/working` artifacts. Update those paths and supply the datasets before running it elsewhere. Its notebook dependencies extend beyond the web application's requirements.

Review the notebook's environment assumptions before rerunning: several cells access `model.module` directly, the final example hard-codes 12 source classes, and the fallback LJSpeech download does not update `REAL_ROOT` to the extracted directory. These experimental cells are not a portable training command.

For web inference, export the trained state dictionary and the exact ordered `all_models` labels using the [setup guide](../docs/setup.md). Do not substitute the example generator labels for the actual trained class mapping. Saved experimental outputs are not a guarantee of performance on new audio.
