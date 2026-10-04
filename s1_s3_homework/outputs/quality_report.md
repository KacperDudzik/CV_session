# Data-quality decisions

Missing values:
- age_approx: impute the train median inside the model pipeline only.
- anatom_site_general: keep a category "unknown". Do not drop the row.
- anatom_site_special: do not use. Almost every value is empty.
- diagnosis_3 and diagnosis_4: leave blank. They are not model inputs.
- melanocytic: do not use. Blank is not a random hole; it mostly means "not marked melanocytic".

Label consistency:
- 5,240 lesions, exactly two images each.
- One image is dermoscopic and the other is clinical: close-up.
- Age, sex, site, and diagnosis fields agree across the pair.

Shortcuts:
- image_type is half and half inside every diagnosis_1 class. It names the view, not the disease.
- image_manipulation is usually "instrument only". Do not use it as an input; it describes the file, not the lesion.
- diagnosis_confirm_type and concomitant_biopsy are the same biopsy decision. Malignant lesions were almost all confirmed by histopathology.

Columns that must not be model inputs, because they leak the label or the archive process:
- diagnosis_2, diagnosis_3, diagnosis_4
- diagnosis_confirm_type, concomitant_biopsy, melanocytic
- isic_id and lesion_id are identifiers, not features
