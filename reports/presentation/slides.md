---
theme: neversink
author: "SUWMIT: Priyam Basu, Jose Cols, Daniel Jarvis, Yongsin Park, Daniel Rodabaugh"
title: "BioLaySumm2025: Instruction-based Summarization with Contrastive Decoding"
transition: slide-left
presenter: dev
drawings:
  persist: false
  syncAll: false
fonts:
  sans: 'Open Sans, Roboto'
  mono: 'Roboto Mono'
  weights: '300,400,600'

layout: cover
class: text-center
color: violet
hideInToc: true
---

# Instruction-based Summarization with Contrastive Decoding

`SUWMIT` Priyam Basu, Jose Cols, Daniel Jarvis, Yongsin Park, Daniel Rodabaugh

<img src="/uw-logo.png" class="w-20 absolute right-10 top-10">

---
layout: top-title
color: violet
---

:: title ::

# Introduction: `BioLaySumm 2025`

:: content ::

  - Biomedical publications contain the **latest** research on **health**-related topics.
  - **Technical language** makes it **difficult** for non-expert audiences to understand their contents.

<v-switch>
  <template #1>

<div class="flex flex-col items-center gap-4 font-mono">

> In temperate climates , winter deaths exceed summer ones . However , there is limited information on the timing and the relative magnitudes of maximum and minimum mortality , by local climate , age group , sex and medical cause of death . We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions .

<solar-arrow-down-bold-duotone class="text-4xl" />

> In the USA , more deaths happen in the winter than the summer . But when deaths occur varies greatly by sex , age , cause of death , and possibly region . Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment . Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed , or whether existing ones are effective .

</div>

  </template>
  <template #2> 

<div class="flex flex-col items-center gap-4 font-mono">

> In temperate climates , winter <span class="neversink-orange-light-scheme ns-c-bind-scheme rounded">deaths exceed</span> summer ones . However , there is <span class="neversink-sky-light-scheme ns-c-bind-scheme rounded">limited information on the timing</span> and the relative magnitudes of maximum and minimum <span class="neversink-sky-light-scheme ns-c-bind-scheme rounded">mortality</span> , <span class="neversink-red-light-scheme ns-c-bind-scheme rounded">by</span> local climate , age group , sex and medical cause of death . We used geo-coded mortality data and wavelets to analyse the <span class="neversink-violet-light-scheme ns-c-bind-scheme rounded">seasonality of mortality</span> by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions .

<solar-arrow-down-bold-duotone class="text-4xl" />

> In the USA , <span class="neversink-orange-light-scheme ns-c-bind-scheme rounded">more deaths</span> happen in the winter than the summer . But <span class="neversink-sky-light-scheme ns-c-bind-scheme rounded">when deaths occur</span> <span class="neversink-red-light-scheme ns-c-bind-scheme rounded">varies greatly</span> by sex , age , cause of death , and possibly region . Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment . Analyzing the <span class="neversink-violet-light-scheme ns-c-bind-scheme rounded">seasonality of deaths</span> can help scientists determine whether interventions to minimize deaths during a certain time of year are needed , or whether existing ones are effective .

</div>

  </template>
  <template #3> 

<div class="flex flex-col items-center gap-4 font-mono">

> In temperate climates , winter deaths exceed summer ones . However , there is limited information on the timing and the relative magnitudes of maximum and minimum mortality , by local climate , age group , sex and medical cause of death . <span class="neversink-amber-light-scheme ns-c-bind-scheme rounded">We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions .</span>

<solar-arrow-down-bold-duotone class="text-4xl" />

> In the USA , more deaths happen in the winter than the summer . But when deaths occur varies greatly by sex , age , cause of death , and possibly region . <span class="neversink-green-light-scheme ns-c-bind-scheme rounded">Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment . Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed , or whether existing ones are effective . </span><span class="neversink-amber-light-scheme ns-c-bind-scheme rounded">Now , Parks et al . show that there are age and sex differences in which times of year most deaths occur .</span>

</div>

  </template>
</v-switch>

---
layout: top-title
color: violet
---

:: title ::

# Introduction: `Data`

:: content ::

- **2 datasets** (`PLOS` and `eLife`) with three splits: `train`, `validation`, and `test`.
- 30,735 biomedical articles with **expert-written** summaries *(English)*. 
- 284 articles (142 per dataset) include only the original text, with **no summaries**.

<v-click>

—

**Observations:**

1. *Readability* and summary length **vary** within each dataset.
2. These variations are significant when **comparing the two** datasets.

</v-click>

---
layout: full
---

Distribution of **summary lengths** across training and validation splits for the `PLOS` and `eLife` datasets

<img src="/length.png" class="h-full w-full object-contain">

---
layout: top-title
color: violet
---

:: title ::

# Related Work

:: content ::

**You et al., 2024**

- TODO


---
layout: top-title
color: violet
---

:: title ::

# Evaluation

:: content ::

10 automated metrics grouped into **3 criteria**:

<v-clicks depth="1">

* **Relevance (4):** ROUGE, BLEU, METEOR, and BERTScore.
  * Measure **lexical overlap** and semantic similarity in embedding space.
* **Readability (4):** Flesch-Kincaid Grade Level, Dale-Chall Readability Score, CLI, and LENS.
  * Combine **surface features** (sentence length) with learned embeddings that correlate with human judgments.
* **Factuality (2):** AlignScore and SummaC.
  * Summary's span alignment with reference and entailment scores to flag contradictions or omissions.

</v-clicks>

---
layout: top-title
color: violet
---

:: title ::

# Evaluation

:: content ::

**Final Score =** Normalize scores by metric, average metrics for each criterion, and average across all criteria.

```mermaid {theme: 'default', nodeSpacing: 10, rankSpacing: 50}
flowchart TB
    R1([ROUGE]) & R2([BLEU]) & R3([METEOR]) & R4([BERTScore]) --> B1[Relevance]
    D1([FKGL]) & D2([DCRS]) & D3([CLI]) & D4([LENS]) --> B2[Readability]
    F1([AlignScore]) & F2([SummaC]) --> B3[Factuality]
    B1 --> AV1((AVG))
    B2 --> AV2((AVG))
    B3 --> AV3((AVG))
    AV1 & AV2 & AV3 --> Score
    
    classDef input fill:none,stroke:#000;
    classDef block fill:#e5e7eb,stroke:#111827;
    class R1,R2,R3,R4,D1,D2,D3,D4,F1,F2 input;
    class B1,B2,B3,AV1,AV2,AV3 block;
    
```

---
layout: top-title
color: violet
---

:: title ::

# Model: `Baseline`

:: content ::

"Extract-then-abstract" pipeline without preprocessing.

<v-clicks>

- **Extract** 40 sentences using `TextRank` and `BioBERT`. 
- Fine-tune Llama 3.1 Instruct using **extractive summaries** as input.

> **Instruction:** You are a specialist medical communicator responsible for translating biomedical articles into a clear, accurate 10–20 sentence summary for non-experts. The summary should be at a Flesch–Kincaid grade level of 10–14 and explain any technical terms.

<div class="mt-4"></div>

```mermaid {theme: 'default'}
flowchart LR
  input([Long Article<br/>+300 sentences])
  extractive["**Extractive** summarizer<br/>TextRank + BioBERT"]
  model("**Fine-tuned** Llama 3.1 Instruct")
  output([**Lay** summary])

  input --> extractive
  extractive -- "40-sentence<br/>summary" --> model
  model --> output

  style input fill:none,stroke:#000;
  style extractive fill:none;
```

</v-clicks>

<v-click>

—

- Comparable performance to You et al., 2024.
  - The `relevance` and `factuality` scores are **slightly worse**.

</v-click>

---
layout: top-title
color: violet
---

:: title ::

# Model: `Interim`

:: content ::

Focus first on improving `factuality` scores.

<v-clicks>

  - **Extractive summaries** alone achieved top scores in factual accuracy.
  - Focusing on 2 metrics is **simpler** than managing 4.
  - Potentially more **impactful** since `factuality` is averaged over 2 metrics rather than 4. 

</v-clicks>

<v-click>

—

**How?**

</v-click>

<v-click>

"Let's just submit the abstracts"

<div class="text-center">

```mermaid {theme: 'default'}
flowchart LR
  input([Long Article])
  extractive[Extract **Abstract**]
  output([Summary])

  input --> extractive
  extractive -- "abstract" --> output

  style input fill:none,stroke:#000;
  style extractive fill:none;
```

</div>

Strong performance in `factuality`: AlignScore of **0.9908** and a SummaC score of **0.9528**.

</v-click>

---
layout: top-title
color: violet
---

:: title ::

# Model: `Interim`

:: content ::

Explore 3 **fine-tuning** experiments that combine the article's **abstract** with other text as input.

<v-click>

- Abstract only:

```mermaid {theme: 'default'}
flowchart LR
  input([Long Article])
  extractive[Extract **Abstract**]
  model(**Fine-tuned** Llama)
  output([**Lay** Summary])

  input --> extractive
  extractive -- "abstract" --> model
  model --> output

  style input fill:none,stroke:#000;
  style extractive fill:none;
```

</v-click>

<v-click>

- Abstract concatenated with extractive summary:

```mermaid {theme: 'default'}
flowchart LR
  input([Long Article])
  extractive[**Extractive** summarizer]
  abstract[Extract **Abstract**]
  concat((\+))
  model(**Fine-tuned** Llama)
  output([**Lay** Summary])

  input --> abstract
  input --> extractive
  abstract -- "abstract" --> concat
  extractive -- "40-sentence<br/>summary" --> concat
  concat --> model
  model --> output

  style input fill:none,stroke:#000;
  style abstract fill:none;
  style extractive fill:none;
```

</v-click>

---
layout: top-title
color: violet
---

:: title ::

# Model: `Interim`

:: content ::

- Abstract concatenated with extractive summary that **excluded** the abstract during extraction.

```mermaid {theme: 'default'}
flowchart LR
  input([Long Article])
  abstract[Extract **Abstract**]
  extractive[**Extractive** summarizer]
  concat((\+))
  remove((\-))
  model(**Fine-tuned** Llama)
  output([**Lay** Summary])

  input -- "full-text" --> remove
  input --> abstract
  abstract -- "abstract" --> remove
  abstract -- "abstract" --> concat
  remove --> extractive
  extractive -- "40-sentence<br/>summary" --> concat
  concat --> model
  model --> output

  style input fill:none,stroke:#000;
  style abstract fill:none;
  style extractive fill:none;
```

<v-click>

—

All these experiments performed **worse** than our baseline.

</v-click>

---
layout: top-title
color: violet
---

:: title ::

# Model: `Final`

:: content ::

TODO

---
layout: full
---

<div class="text-center">

Final **pipeline** for rapid experimentation in summarization

</div>

<img src="/pipeline.png" class="h-full w-full object-contain pb-10 pt-4">

---
layout: top-title
color: violet
---

:: title ::

# Results

:: content ::

- TODO

---
layout: top-title
color: violet
---

:: title ::

# Discussion

:: content ::

- TODO

---
layout: center
---

# Questions?

Parts of this work was/were completed on Hyak, UW's high-performance computing cluster. This resource was funded by the Student Technology Fee.