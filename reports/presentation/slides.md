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

**Subtask 1.1** - Plain lay summarization.

  - Biomedical publications contain the **latest** research on **health**-related topics.
  - **Technical language** makes it **difficult** for non-expert audiences to understand their contents.

<v-switch>
  <template #1>

<div class="flex flex-col items-center gap-4 font-mono">

> **Article:** In temperate climates , winter deaths exceed summer ones . However , there is limited information on the timing and the relative magnitudes of maximum and minimum mortality , by local climate , age group , sex and medical cause of death . We used geo-coded mortality data and wavelets to analyse the seasonality of mortality by age group and sex from 1980 to 2016 in the USA and its subnational climatic regions .

<solar-arrow-down-bold-duotone class="text-4xl" />

> **Summary:** In the USA , more deaths happen in the winter than the summer . But when deaths occur varies greatly by sex , age , cause of death , and possibly region . Seasonal differences in death rates can change over time due to changes in factors that cause disease or affect treatment . Analyzing the seasonality of deaths can help scientists determine whether interventions to minimize deaths during a certain time of year are needed , or whether existing ones are effective .

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

1. *Readability* and *summary length* **vary** within each dataset.
2. These variations are **significant** when comparing the two datasets.

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

**Turbitt et al., 2023**

- Winners of BioLaySumm 2023.
- Few-shot prompting using the `text-davinci-003` model from OpenAI.
- A fine-tuned BioGPT model, 1/100th the size of `text-davinci-003`, showed comparable performance.

<v-click>

**You et al., 2024**

- Winners of BioLaySumm 2024.
- Extract-then-Summarize method with a fine-tuned GPT-3.5 model.
- They found that including external knowledge was detrimental for `factuality`.  

</v-click>

---
layout: top-title
color: violet
---

:: title ::

# Related Work

:: content ::

**Modi and Karthikeyan, 2024**

- Best `factuality` scores at BioLaySumm 2024.
- **Preprocess abstracts** to remove content within parentheses, braces, and brackets.

<v-click>

**Ribeiro et al., 2023**

- 3 generation techniques for fine-grained control over the **readability** of summaries.
- Instruction-based readability control: *"Summarize this for a middle school student."*

</v-click>

<v-click>

**Chuang et al., 2024**

- Decoding strategy (**DoLa**) that generates the next-token distribution by contrasting the differences in logits across specific layers.
- Improved **factual** accuracy in question-answering tasks.

</v-click>

---
layout: top-title
color: violet
---

:: title ::

# Evaluation: `Metrics`

:: content ::

10 automated metrics grouped into **3 criteria**:

<v-clicks depth="1">

* **Relevance (4):** ROUGE, BLEU, METEOR, and BERTScore.
  * Measure **lexical overlap** and **semantic similarity** in embedding space.
* **Readability (4):** Flesch-Kincaid Grade Level, Dale-Chall Readability Score, CLI, and LENS.
  * Combine **surface features** (sentence length) with **learned embeddings** that correlate with human judgments.
* **Factuality (2):** AlignScore and SummaC.
  * Summary's span alignment with reference and entailment scores to flag contradictions or omissions.

</v-clicks>

---
layout: top-title
color: violet
---

:: title ::

# Evaluation: `Final Score`

:: content ::

(1) Normalize scores by metric, (2) average metrics for each criterion, and (3) average across all criteria.

<div class="my-10"></div>

```mermaid {theme: 'default', nodeSpacing: 10, rankSpacing: 50}
flowchart TB
    R1([ROUGE]) & R2([BLEU]) & R3([METEOR]) & R4([BERTScore]) --> AV1((AVG))
    D1([FKGL]) & D2([DCRS]) & D3([CLI]) & D4([LENS]) --> AV2((AVG))
    F1([AlignScore]) & F2([SummaC]) --> AV3((AVG))
    AV1 --> B1[Relevance]
    AV2 --> B2[Readability]
    AV3 --> B3[Factuality]
    B1 & B2 & B3 --> AV4((AVG))
    AV4 --> Score
    
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

"Extract-then-abstract" pipeline.

<v-clicks>

- **Extract** 40 sentences using `TextRank` and `BioBERT`. 
- Fine-tuned Llama 3.1 Instruct (8B) using **extractive summaries** as input and **DoLa** for decoding.

> **Instruction:** You are a specialist medical communicator responsible for translating biomedical articles into a clear, accurate 10–20 sentence summary for non-experts. The summary should be at a <ins>Flesch–Kincaid grade level of 10–14</ins> and explain any technical terms.

<div class="mt-4"></div>

```mermaid {theme: 'default'}
flowchart LR
  input([Article<br/>+300 sentences])
  extractive["**Extractive** Summarizer<br/>TextRank + BioBERT"]
  model("**Fine-tuned** Llama")
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

  - **Extractive summaries** alone achieved better scores `factuality` scores.
  - Focusing on 2 metrics is **simpler** than managing 4.
  - Potentially more **impactful** since `factuality` is averaged over 2 metrics rather than 4. 

</v-clicks>

<v-click>

—

**How?**

</v-click>

<v-click>

Previous research shows that <span v-mark.underline.red v-click="+0">abstracts</span> are highly relevant to the ground truth summary. 

<div class="text-center">

```mermaid {theme: 'default'}
flowchart LR
  input([Article])
  extractive[**Abstract** Extraction]
  output([Summary])

  input --> extractive
  extractive -- "abstract" --> output

  style input fill:none,stroke:#000;
  style extractive fill:none;
```

</div>

<v-click>

Strong performance in `factuality`: AlignScore of **0.9908** and a SummaC score of **0.9528**.

</v-click>

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

<span class="font-serif font-bold">(I)</span> Abstract only:

```mermaid {theme: 'default'}
flowchart LR
  input([Article])
  extractive[**Abstract** Extraction]
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

<span class="font-serif font-bold">(II)</span> Abstract concatenated with extractive summary:

```mermaid {theme: 'default'}
flowchart LR
  input([Article])
  extractive[**Extractive** Summarizer]
  abstract[**Abstract** Extraction]
  concat((\+))
  model(**Fine-tuned** Llama)
  output([**Lay** Summary])

  input --> abstract
  input --> extractive
  abstract -- "abstract" --> concat
  extractive -- "extractive summary" --> concat
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

<span class="font-serif font-bold">(III)</span> Abstract concatenated with extractive summary that **excluded** the abstract during extraction.

```mermaid {theme: 'default', nodeSpacing:10, rankSpacing:30}
flowchart LR
  input([Article])
  abstract[**Abstract** Extraction]
  extractive[**Extractive** Summarizer]
  concat((\+))
  remove((\-))
  model(**Fine-tuned** Llama)
  output([**Lay** Summary])

  input -- "full-text" --> remove
  input --> abstract
  abstract -- "abstract" --> remove
  abstract -- "abstract" --> concat
  remove --> extractive
  extractive -- "extractive summary" --> concat
  concat --> model
  model --> output

  style input fill:none,stroke:#000;
  style abstract fill:none;
  style extractive fill:none;
```

---
layout: top-title
color: violet
transition: view-transition
---

:: title ::

# Model: `Interim`

:: content ::

<div class="text-center">

Metric scores of the **interim** experiments on `eLife` validation

</div>

<div class="rounded overflow-clip border border-zinc-200">
<table class="table-auto text-xs">
    <thead>
    <tr class="divide-x divide-zinc-300 bg-zinc-100">
        <th class="text-center">MODEL</th>
        <th class="text-center">K</th>
        <th class="text-center" colspan="4">RELEVANCE</th>
        <th class="text-center" colspan="4">READABILITY</th>
        <th class="text-center" colspan="2">FACTUALITY</th>
    </tr>
    <tr class="bg-zinc-50">
        <th></th>
        <th></th>
        <th class="font-light">ROUGE</th>
        <th class="font-light">BLEU</th>
        <th class="font-light">METEOR</th>
        <th class="font-light">BertS</th>
        <th class="font-light">FKGL</th>
        <th class="font-light">DCRS</th>
        <th class="font-light">CLI</th>
        <th class="font-light">LENS</th>
        <th class="font-light">AlignS</th>
        <th class="font-light">SummaC</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>Our Baseline</td>
        <td>40</td>
        <td>0.3792</td>
        <td>8.4205</td>
        <td>0.2852</td>
        <td><strong>0.8568</strong></td>
        <td>9.0037</td>
        <td><strong>7.5340</strong></td>
        <td>10.0081</td>
        <td>78.4724</td>
        <td>0.6433</td>
        <td>0.6453</td>
    </tr>
    <tr>
        <td><span class="font-serif font-bold">(I)</span> Abs</td>
        <td>--</td>
        <td>0.3694</td>
        <td>7.5316</td>
        <td>0.2773</td>
        <td>0.8541</td>
        <td><strong>8.7827</strong></td>
        <td>10.2781</td>
        <td><strong>9.8029</strong></td>
        <td><strong>79.4479</strong></td>
        <td>0.6339</td>
        <td><strong>0.6632</strong></td>
    </tr>
    <tr>
        <td><span class="font-serif font-bold">(II)</span> Abs+Ext</td>
        <td>40</td>
        <td><strong>0.3818</strong></td>
        <td><strong>8.6506</strong></td>
        <td><strong>0.2966</strong></td>
        <td>0.8552</td>
        <td>8.9564</td>
        <td>10.2320</td>
        <td>9.9339</td>
        <td>76.6738</td>
        <td><strong>0.6458</strong></td>
        <td>0.6082</td>
    </tr>
    <tr>
        <td><span class="font-serif font-bold">(III)</span> Abs+Ext(abs)</td>
        <td>30</td>
        <td>0.3717</td>
        <td>8.1089</td>
        <td>0.2842</td>
        <td>0.8537</td>
        <td>9.0198</td>
        <td>10.3763</td>
        <td>9.9745</td>
        <td>78.0254</td>
        <td>0.6428</td>
        <td>0.6433</td>
    </tr>
    </tbody>
</table>
</div>

---
layout: top-title
color: violet
---

:: title ::

# Model: `Interim`

:: content ::

<div class="text-center">

**Normalized** scores of the **interim** experiments on `eLife` validation *(sorted)*

</div>

<div class="rounded overflow-clip border border-zinc-200">
<table class="table-auto text-xs">
    <thead>
    <tr class="divide-x divide-zinc-300 bg-zinc-100">
        <th class="text-center">MODEL</th>
        <th class="text-center">K</th>
        <th class="text-center">RELEVANCE</th>
        <th class="text-center">READABILITY</th>
        <th class="text-center">FACTUALITY</th>
        <th class="text-center">SCORE</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>Our Baseline</td>
        <td>40</td>
        <td>0.906905</td>
        <td><strong>0.958134</strong></td>
        <td>0.361242</td>
        <td><strong>0.742094</strong></td>
    </tr>
    <tr>
        <td><span class="font-serif font-bold">(II)</span> Abs+Ext</td>
        <td>40</td>
        <td><strong>0.938622</strong></td>
        <td>0.773277</td>
        <td>0.319408</td>
        <td>0.677102</td>
    </tr>
    <tr>
        <td><span class="font-serif font-bold">(III)</span> Abs+Ext(abs)</td>
        <td>30</td>
        <td>0.856543</td>
        <td>0.764911</td>
        <td>0.358361</td>
        <td>0.659939</td>
    </tr>
    <tr>
        <td><span class="font-serif font-bold">(I)</span> Abs</td>
        <td>--</td>
        <td>0.815440</td>
        <td>0.789736</td>
        <td><strong>0.372712</strong></td>
        <td>0.659296</td>
    </tr>
    </tbody>
</table>
</div>

<v-click>

**Observations:**

1. All our interim experiments performed **worse** than our baseline.
2. **Longer inputs** tend to perform better.
3. Repeating relevant information appears to be helpful.

</v-click>

---
layout: top-title
color: violet
---

:: title ::

# Model: `Final`

:: content ::

Fine-tuned Llama 3.1 Instruct (8B) using the **entire article** text as input.

<v-clicks>

- **No** preprocessing or post-processing.
- Apply Decoding by Contrasting Layers (**DoLa**) during inference.
- Trained **one model per dataset** (`train` + `validation` splits), using a total of 30,735 articles.

</v-clicks>

<div class="text-center" v-click>

```mermaid {theme: 'default'}
flowchart LR
  input([Article])
  model("**Fine-tuned** Llama")
  output([**Lay** summary])

  input --> model
  model --> output

  style input fill:none,stroke:#000;
```

</div>

<v-click>

**Hyperparameters**

- LoRA: $\text{rank}=8$, $\alpha=16$, $\text{dropout}=0.0$
- Training: $\text{optimizer}=\text{adamw}$, $\alpha=3e^{-4}$, $\text{epoch}=2$
- Inference: $\text{max\_new\_tokens}=384$, $\text{batch\_size}=1$, $\text{dola\_layers}=\{0,2,20\}$

</v-click>

---
layout: full
---

<div class="text-center">

Final **pipeline** for rapid experimentation in summarization

</div>

<img src="/pipeline.png" class="h-full w-full object-contain pb-20 pt-4">

---
layout: top-title
color: violet
transition: view-transition
---

:: title ::

# Results

:: content ::

<div class="text-center">

Metric scores of the **final** experiments on `eLife` validation

</div>

<div class="rounded overflow-clip border border-zinc-200">
<table class="table-auto text-xs">
    <thead>
    <tr class="divide-x divide-zinc-300 bg-zinc-100">
        <th class="text-center">MODEL</th>
        <th class="text-center">K</th>
        <th class="text-center" colspan="4">RELEVANCE</th>
        <th class="text-center" colspan="4">READABILITY</th>
        <th class="text-center" colspan="2">FACTUALITY</th>
    </tr>
    <tr class="bg-zinc-50">
        <th></th>
        <th></th>
        <th class="font-light">ROUGE</th>
        <th class="font-light">BLEU</th>
        <th class="font-light">METEOR</th>
        <th class="font-light">BertS</th>
        <th class="font-light">FKGL</th>
        <th class="font-light">DCRS</th>
        <th class="font-light">CLI</th>
        <th class="font-light">LENS</th>
        <th class="font-light">AlignS</th>
        <th class="font-light">SummaC</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>Baseline</td>
        <td>40</td>
        <td>0.3792</td>
        <td>8.4205</td>
        <td>0.2852</td>
        <td>0.8568</td>
        <td>9.0037</td>
        <td><strong>7.5340</strong></td>
        <td>10.0081</td>
        <td>78.4724</td>
        <td>0.6433</td>
        <td>0.6453</td>
    </tr>
    <tr>
        <td>Abs</td>
        <td>--</td>
        <td>0.3694</td>
        <td>7.5316</td>
        <td>0.2773</td>
        <td>0.8541</td>
        <td>8.7827</td>
        <td>10.2781</td>
        <td><strong>9.8029</strong></td>
        <td><strong>79.4479</strong></td>
        <td>0.6339</td>
        <td><strong>0.6632</strong></td>
    </tr>
    <tr>
        <td>Abs<sub>pre</sub></td>
        <td>--</td>
        <td>0.3727</td>
        <td>8.1257</td>
        <td><strong>0.2890</strong></td>
        <td>0.8532</td>
        <td><strong>8.7326</strong></td>
        <td>10.2495</td>
        <td>9.8086</td>
        <td>77.5272</td>
        <td>0.6370</td>
        <td>0.5990</td>
    </tr>
    <tr class="bg-yellow-100">
        <td>Full-text</td>
        <td>--</td>
        <td><strong>0.3851</strong></td>
        <td><strong>8.6941</strong></td>
        <td>0.2887</td>
        <td><strong>0.8591</strong></td>
        <td>9.3079</td>
        <td>7.6741</td>
        <td>10.1434</td>
        <td>78.6703</td>
        <td>0.6432</td>
        <td>0.6629</td>
    </tr>
    <tr>
        <td>Full-text<sub>post</sub></td>
        <td>--</td>
        <td>0.3842</td>
        <td>8.5227</td>
        <td>0.2867</td>
        <td>0.8587</td>
        <td>9.3294</td>
        <td>10.4553</td>
        <td>10.1530</td>
        <td>79.2063</td>
        <td><strong>0.6444</strong></td>
        <td>0.6616</td>
    </tr>
    </tbody>
</table>
</div>

---
layout: top-title
color: violet
---

:: title ::

# Results

:: content ::

<div class="text-center">

**Normalized** scores of the **final** experiments on `eLife` validation *(sorted)*

</div>

<div class="rounded overflow-clip border border-zinc-200">
<table class="table-auto text-xs">
    <thead>
    <tr class="divide-x divide-zinc-300 bg-zinc-100">
        <th class="text-center">MODEL</th>
        <th class="text-center">K</th>
        <th class="text-center">RELEVANCE</th>
        <th class="text-center">READABILITY</th>
        <th class="text-center">FACTUALITY</th>
        <th class="text-center">SCORE</th>
    </tr>
    </thead>
    <tbody>
    <tr class="bg-yellow-100">
        <td>Full-text</td>
        <td>--</td>
        <td><strong>0.954173</strong></td>
        <td>0.935548</td>
        <td><strong>0.382233</strong></td>
        <td><strong>0.757318</strong></td>
    </tr>
    <tr>
        <td>Our Baseline</td>
        <td>40</td>
        <td>0.906905</td>
        <td><strong>0.958134</strong></td>
        <td>0.361242</td>
        <td>0.742094</td>
    </tr>
    <tr>
        <td>Full-text<sub>post</sub></td>
        <td>--</td>
        <td>0.937714</td>
        <td>0.748157</td>
        <td>0.381944</td>
        <td>0.689272</td>
    </tr>
    <tr>
        <td>Abs</td>
        <td>--</td>
        <td>0.815440</td>
        <td>0.789736</td>
        <td>0.372712</td>
        <td>0.659296</td>
    </tr>
    <tr>
        <td>Abs<sub>pre</sub></td>
        <td>--</td>
        <td>0.868276</td>
        <td>0.786174</td>
        <td>0.299148</td>
        <td>0.651199</td>
    </tr>
    </tbody>
</table>
</div>

<v-click>

**Observations:**

1. The **full-text** model trades lower ↓`readability` for better ↑`factuality`.
2. Preprocessing and post-processing **decreased** performance.

</v-click>

---
layout: top-title
color: violet
---

:: title ::

# Results

:: content ::

<div class="text-center">

**Full-text** model scores using different **decoding** strategies on the validation splits *(sorted)*

</div>

<div class="rounded overflow-clip border border-zinc-200">
<table class="table-auto text-xs">
    <thead>
    <tr class="divide-x divide-zinc-300 bg-zinc-100">
        <th class="text-center">MODEL</th>
        <th class="text-center">RUNTIME</th>
        <th class="text-center">RELEVANCE</th>
        <th class="text-center">READABILITY</th>
        <th class="text-center">FACTUALITY</th>
        <th class="text-center">SCORE</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>DoLa</td>
        <td>40</td>
        <td>0.906905</td>
        <td><strong>0.958134</strong></td>
        <td>0.361242</td>
        <td><strong>0.742094</strong></td>
    </tr>
    <tr>
        <td>Greedy decoding</td>
        <td>40</td>
        <td><strong>0.938622</strong></td>
        <td>0.773277</td>
        <td>0.319408</td>
        <td>0.677102</td>
    </tr>
    <tr>
        <td>Beam search</td>
        <td>30</td>
        <td>0.856543</td>
        <td>0.764911</td>
        <td>0.358361</td>
        <td>0.659939</td>
    </tr>
    </tbody>
</table>
</div>

---
layout: top-title
color: violet
---

:: title ::

# Conclusions

:: content ::

- TODO

---
layout: center
---

# Questions?

Parts of this work were completed on Hyak, UW's high-performance computing cluster. This resource was funded by the Student Technology Fee.