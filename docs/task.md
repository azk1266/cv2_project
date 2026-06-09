# Computer Vision II Final Project Brief

**Computer Vision II · Master in Artificial Intelligence · Universidade de Vigo · ESEI**

**Academic year 2025–26**

## 1. Overview

This project asks you to work as a researcher. The central task is to identify a specific open problem in a domain of your choice, find the most promising current approach to that problem in the recent literature, and evaluate whether it makes progress on it using domain-specific data.

You are not constrained to architectures covered in the course. The course content gives you the vocabulary and conceptual grounding to read and evaluate recent papers. Just use that foundation to go further! The architecture you implement may be a 2023 or 2024 model not discussed in any lecture, provided you can justify its selection from first principles.

The sequence matters: literature first, challenge second, architecture third, experiments fourth. Skipping or reordering this sequence produces a weaker project and will probably result in a lower-quality work.

## 2. Required Workflow

**Step 1. Literature first.** Read the abstract, introduction, and open problems sections of the reference survey for your chosen domain. Identify one open challenge (a concrete unsolved problem with a measurable gap).

**Step 2. Research question.** Formulate a research question: can architecture X address challenge Y on data type Z? State an initial hypothesis.

**Step 3. Architecture selection.** Search the recent literature (2022–2025) for current approaches to the challenge. Try to select the architecture whose design properties best address it. Write a justification that names at least one alternative you considered and explains why you rejected it. Benchmark accuracy alone is not a justification.

**Step 4. Experiments.** Acquire a dataset, implement the pipeline using a pretrained or publicly available model, and run baseline experiments. Compute at least two quantitative metrics on a held-out test set.

**Step 5. Failure analysis.** Identify at least two failure cases and explain them mechanistically, grounded in the architecture's design.

**Step 6. Extension [OPTIONAL].** Implement one modification motivated by the failure analysis and measure its effect. Report the result whether or not it improves performance.

**Step 7. Report.** Write the report. The thesis direction section must connect your results to your initial hypothesis and propose one concrete next step.

## 3. Worked Example

The following example illustrates in a summarized way what Steps 1–3 look like. It is drawn from Domain A but the structure applies to all domains. Your own project does not need to follow this topic; it is merely used to illustrate the level of specificity and the type of reasoning expected.

### Worked Example — Domain A

**Step 1 — Challenge identified from the survey:**

Li et al. (2024) identify long-range temporal dependency modelling as a major open problem in surgical phase recognition. Current clip-based models process fixed windows of 8–32 frames and aggregate predictions independently, which means they cannot use information from the beginning of an operation to inform predictions at the end. On long procedures (Cholec80 average duration: 38 minutes), this causes systematic errors at phase transitions where the model lacks the context needed to distinguish, for example, clipping from dissection.

**Step 2 — Research question and hypothesis:**

Can a transformer-based video model with global self-attention reduce phase transition errors on Cholec80 compared to a clip-based 3D CNN baseline? Hypothesis: yes, because self-attention can model dependencies across the full sequence without the fixed-window constraint.

**Step 3 — Architecture selection from recent literature:**

A search of recent work on long-range video understanding identifies three candidates:

1. A SlowFast network with an LSTM aggregation head
2. TimeSformer with factorized attention
3. Video-Swin Transformer with shifted window attention

SlowFast is rejected because the LSTM still processes clips sequentially and accumulates error over long sequences. TimeSformer factorizes space and time independently, which reduces memory but may lose spatiotemporal correlations relevant to tool-tissue interaction. Video-Swin is selected because its shifted window mechanism allows it to model local spatiotemporal patterns while still enabling cross-window information flow across the full sequence, directly addressing the long-range dependency problem without the quadratic memory cost of full attention.

Note what this example does not do:

- It does not say 'Video-Swin achieves state-of-the-art accuracy on Kinetics-400' as a justification.
- It does not choose an architecture first and then look for a challenge to fit it.
- It does not use a course architecture (SlowFast) simply because it is familiar. It considers it and rejects it for a specific reason.

## 4. Project Domains

Select one domain. The open challenges listed are starting points drawn from the reference surveys. You may identify a different challenge within the domain provided it is equally specific and grounded in the literature.

### Domain C

**Gesture and Sign Language Recognition**

Apply video classification or tracking to hand gesture or sign language for HCI applications.

**Open challenges**

- Continuous sign language recognition — segmenting and classifying unsegmented sign streams
- Cross-signer generalization — models trained on one signer fail on others
- Low-resource sign language recognition with limited labelled data
- Real-time gesture recognition with low-latency constraints on edge devices

**Datasets**

WLASL (Word-Level American Sign Language), AUTSL, EgoGesture, NVGesture.

**Surveys**

Zhou et al. (2023). Human pose-based estimation, tracking and action recognition. arXiv:2310.13039

**Lessons**

Course connection: Lessons 02 (action recognition, two-stream networks), 03 (tracking), 04 (depth-based pose estimation)

## 5. Deliverables

Students are required to produce two assets as the result of their work for the final project:

- **Notebook.** A working Jupyter notebook (.ipynb) hosted on Kaggle. The notebook must run end-to-end without errors using only the attached dataset and pretrained models.
- **Report.** A technical report of 5–6 pages (excluding references) following the structure in Section 7. The report file might use any of the following formats: pdf, doc or docx.

File submission is not allowed. Instead, submit links pointing to the two required deliverables via the course’s submission form before the deadline indicated in the academic calendar (June 9).

Submissions missing either item will not be graded.

In those cases where two students are working on the same project, both need to submit their solution through the form.

## 6. Technical Requirements

### 6.1 Architecture

You are not limited to architectures covered in the course. Select the most appropriate recent model for your challenge. You may use any pretrained model from a public repository (HuggingFace, GitHub, PyTorch Hub). Training from scratch is not required. If you fine-tune, document the training configuration precisely so results are reproducible.

The architecture justification (Section 7.3 of the report) is the most heavily weighted component. A strong justification connects the architecture's specific design properties to the challenge, considers at least one alternative, and explains the rejection. Benchmark accuracy is not a justification.

### 6.2 Evaluation

Compute at least two quantitative metrics on a held-out test set appropriate for your task. For classification: accuracy and at least one of F1, precision, recall, or AUC-ROC. For tracking: mean IoU and AUC of the success curve following the OPE protocol from Lab 03. For regression: RMSE and at least one scale-invariant metric. Do not report training metrics as your main results.

### 6.3 Failure analysis

Identify at least two failure cases. For each: show a visual example, state the metric value, and provide a mechanistic explanation. A mechanistic explanation connects the failure to a specific design property or assumption of the architecture, not to generic difficulty.

### 6.4 Extension experiment [OPTIONAL]

Implement one modification motivated by the failure analysis and addressing a specific weakness identified in Section 6.3. Measure and report the effect. Negative results are valid and will be graded on the quality of the analysis, not the outcome.

## 7. Report Structure

The report must follow this structure. The sequence reflects the research workflow: problem first, architecture second, results third. Do not reorder sections.

| Section                   | Pages | Content                                                                                                               |
| ------------------------- | ----: | --------------------------------------------------------------------------------------------------------------------- |
| 1. Problem statement      |  0.75 | Open challenge from the survey; why it matters; research question; initial hypothesis                                 |
| 2. Related work           |   0.5 | Key papers on the challenge; what has been tried; where the gap remains                                               |
| 3. Architecture selection |  0.75 | Chosen model and its source; why its design addresses the challenge; at least one alternative considered and rejected |
| 4. Experimental setup     |   0.5 | Dataset; evaluation protocol; metrics; preprocessing                                                                  |
| 5. Results                |     1 | Quantitative results; visualizations; comparison if applicable                                                        |
| 6. Failure analysis       |  0.75 | Two failure cases with mechanistic explanations grounded in architecture design                                       |
| 7. Thesis direction       |  0.75 | Did results support or refute the hypothesis? What would need to change? One concrete next step with proposed method  |
| References                |   0.5 | Around 10 peer-reviewed sources                                                                                       |

The thesis direction section is not a future work section. It must answer: Did your results support or refute your initial hypothesis? What does the failure analysis reveal about what would need to change? What is one concrete next step with a proposed method and a feasibility argument?

## 8. Evaluation Criteria

| Component                                             | Weight | Key criterion                                                                                       |
| ----------------------------------------------------- | -----: | --------------------------------------------------------------------------------------------------- |
| Problem identification and architecture justification |    30% | Challenge is specific; architecture choice follows from it mechanistically; alternatives considered |
| Experimental results                                  |    20% | Metrics computed correctly on a held-out test set                                                   |
| Failure analysis                                      |    20% | Failures explained mechanistically; grounded in architecture design                                 |
| Thesis direction                                      |    20% | Hypothesis revised in light of results; next step is concrete and feasible                          |
| Report quality                                        |    10% | Clear, concise, active voice, correctly referenced                                                  |

A student who identifies a real challenge, selects an architecture for principled reasons, honestly evaluates whether it works, and revises their understanding accordingly has demonstrated research competence regardless of whether the hypothesis is confirmed. A student who selects an architecture first and retrofits a justification has not.

## 9. Suggested Timeline

| Week   |  Hours | Activities                                                                                                                 |
| ------ | -----: | -------------------------------------------------------------------------------------------------------------------------- |
| Week 1 |  6–8 h | Read survey → identify challenge → search recent literature → select architecture → formulate hypothesis → acquire dataset |
| Week 2 | 8–10 h | Implement pipeline → baseline experiments → compute metrics → failure analysis → extension experiment                      |
| Week 3 |  3–4 h | Analyze results vs hypothesis → revise thesis direction → write report                                                     |

## 10. Academic Integrity

You may use any publicly available pretrained model, library, or code, provided you cite it. You may not submit work produced collaboratively with another student. The problem identification, architecture justification, failure analysis, and thesis direction must reflect your own reasoning.

If you use an LLM to assist with code generation or writing, declare it explicitly in the notebook or report. Undeclared LLM use that is detected will be treated as academic misconduct.

## 11. Reference Surveys by Domain

Read the open problems and future directions sections of the relevant survey before defining your research question. Your challenge must be grounded in what the survey identifies as unsolved — not a problem already addressed in the literature.

### Domain C — Gesture and Sign Language

- Zhou L. et al. (2023). Human pose-based estimation, tracking and action recognition with deep learning: a survey. arXiv:2310.13039
- Seweryn K. et al. (2023). Survey of action recognition, spotting, and spatio-temporal localization in soccer. ACM Computing Surveys. DOI: 10.1145/3776541
