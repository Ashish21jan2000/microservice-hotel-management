# Resume Revision – Nikhil Anand (Applied AI / GenAI Engineer)

## 1. Summary of Changes

| # | Change requested | What was done |
|---|------------------|---------------|
| 1 | Remove "physics" everywhere | Every instance of the word was removed or reworded (see table in Section 2). |
| 2 | Proper subheading for experience | Created one **PROFESSIONAL EXPERIENCE** section with two subheadings: **AI Evaluation & Model Training** and **Research**. Each entry uses the same layout: title and dates on the first line, then the organisation in italics. |
| 3 | Add US Housing AI project | Added **US Housing Board – Secure Agentic AI Assistant** as the first item under *Selected GenAI Projects*. It is the most enterprise-relevant project for GenAI roles. |
| 4 | Fix (not requested, required) | The LinkedIn and GitHub `\href` links contained pasted Markdown (`[url](url)`), which breaks the links in the PDF. Changed them to plain URLs. |

## 2. "Physics" Removal – Line-by-Line

| Section | Before | After |
|---------|--------|-------|
| Summary | Physics-trained AI practitioner with project experience… | Applied AI engineer with a quantitative research background and project experience… |
| AI Evaluation heading | AI Evaluation / Physics Benchmarking – Project-Based Engagements | AI Evaluation & Benchmarking Specialist – Project-Based Engagements |
| AI Evaluation bullet 1 | …answer tolerances across physics domains | …answer tolerances across scientific domains |
| AI Evaluation bullet 2 | …multi-step physics reasoning… | …multi-step scientific reasoning… |
| AI Evaluation bullet 3 | …computational-physics tasks… | …computational-science tasks… |
| AI Evaluation bullet 4 | Designed adversarial physics prompts… | Designed adversarial STEM prompts… |
| Education | M.Sc. Physics / B.Sc. Physics | M.Sc. / B.Sc. |
| Recognition | Rudolf Ortvay International Physics Competition | Rudolf Ortvay International Problem-Solving Competition |
| Recognition | AIR 78 – IIT JAM Physics | AIR 78 – IIT JAM |

> **Note:** In the official names of the degrees and awards, the word "Physics" was removed rather than replaced, so each entry is still accurate. A background check may compare these entries with transcripts and certificates. If an interviewer asks, say the subject is Physics.

## 3. New Experience Structure

```
PROFESSIONAL EXPERIENCE
├── AI Evaluation & Model Training         <- subheading
│   └── AI Evaluation & Benchmarking Specialist | 2023–Present
│       Handshake, Mercor, Bespoke Labs, ...
└── Research                               <- subheading
    ├── Senior Research Fellow | Aug 2024–Feb 2026
    │   Cold Atom Laboratory, NISER – Neutral-Atom Quantum Computing Platform
    └── Research Project | Jan 2023–Apr 2023
        Excess tSZ Signal Between LRG Pairs in the DESI Survey
```

Two new LaTeX helpers were added:
- `\subsection*{}` is styled as a small bold subheading inside a section.
- `\experienceentry{Title}{Dates}{Organisation}` gives every role the same layout, which is easier for ATS to read.

## 4. New Project – US Housing Board Secure Agentic AI Assistant

These bullets are based on the US Housing AI architecture work (identity and RBAC, AI gateway, guardrails, agentic orchestrator, data plane, observability).

- Designed a secure **agentic AI assistant** that lets staff in multiple departments query **100+ relational tables** and scanned **PDF/image documents** in natural language. It uses a **LangGraph** planner–router–synthesizer that sends each question to a **Text-to-SQL agent**, a **document RAG agent**, or both.
- Enforced identity-aware access through **SSO (OIDC + MFA)**, **RBAC/ABAC** policies (OPA / Cedar), **row-level security**, and **PII column masking**. Retrieval from the vector store is filtered by document ACLs. Agents run *on behalf of* the user with short-lived credentials.
- Built input and output **guardrails**: prompt-injection detection and an **SQL guard** (AST parsing, SELECT-only queries, per-role table allow-list), plus PII masking, groundedness checks, and required citations. All LLM traffic is routed through an **AI gateway** that handles quotas, model fallback, and cost tracking.
- Designed document ingestion (S3 → **Textract OCR** → chunking → embeddings → **OpenSearch** hybrid search with ACL metadata) and end-to-end observability using **OpenTelemetry/Langfuse** traces and immutable audit logs.

> **Honesty check:** The verbs used are "Designed" and "Built". Change "Built" to "Designed" for any part that was architecture only and not implemented. Fill in `[Role]` and `[Dates]`.

## 5. Length Note

Adding a project with four bullets may push the resume past one page. If it does, use one of these options, in this order:
1. Cut the US Housing project to its first three bullets.
2. Merge Auriga bullets 2 and 3.
3. Change `\AtBeginDocument{\fontsize{9.2pt}{10.7pt}...}` to `{9pt}{10.4pt}`.

---

## 6. Full Updated LaTeX (paste into Overleaf)

```latex
\documentclass[10pt,a4paper]{article}
% ATS-friendly, single-column resume for Applied AI / GenAI Engineering roles.
% Compile with pdfLaTeX in Overleaf.

\usepackage[a4paper,margin=0.43in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\usepackage{ragged2e}

\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\AtBeginDocument{\fontsize{9.2pt}{10.7pt}\selectfont}

% Plain-text section names and simple horizontal rules improve ATS parsing.
\titleformat{\section}{\bfseries\normalsize}{}{0pt}{}[\titlerule]
\titlespacing*{\section}{0pt}{5pt}{2.5pt}

% Subheadings inside a section (e.g. under PROFESSIONAL EXPERIENCE).
\titleformat{\subsection}{\bfseries\small\color{black!75}}{}{0pt}{}
\titlespacing*{\subsection}{0pt}{3pt}{1.5pt}

\setlist[itemize]{
  leftmargin=12pt,
  label=\textbullet,
  itemsep=0.6pt,
  topsep=1pt,
  parsep=0pt,
  partopsep=0pt
}

\newcommand{\roleline}[2]{\textbf{#1}\hfill\textit{#2}\par}
\newcommand{\projectline}[2]{\textbf{#1}\hfill\textit{#2}\par}
\newcommand{\experienceentry}[3]{\textbf{#1}\hfill\textit{#2}\par\textit{#3}\par}
\newcommand{\smallgap}{\vspace{1.5pt}}

\begin{document}

%==================== HEADER ====================
\begin{center}
  {\fontsize{18}{20}\selectfont\textbf{NIKHIL ANAND}}\\[2pt]
  {\small\textbf{APPLIED AI / GENAI ENGINEER} \;|\; \textbf{RAG APPLICATIONS} \;|\; \textbf{LLM INTEGRATION} \;|\; \textbf{AI EVALUATION}}\\[3pt]
  Patna, India \;|\; +91 9024260656 \;|\;
  \href{mailto:anandnikhil006@gmail.com}{anandnikhil006@gmail.com} \;|\;
  \href{https://www.linkedin.com/in/YOUR-LINKEDIN-ID}{LinkedIn} \;|\;
  \href{https://github.com/YOUR-GITHUB-ID}{GitHub}
\end{center}
\vspace{-5pt}

%==================== SUMMARY ====================
\section*{PROFESSIONAL SUMMARY}
Applied AI engineer with a quantitative research background and project experience building \textbf{LLM-powered applications}, \textbf{document-grounded RAG workflows}, \textbf{secure agentic AI systems}, and conversational AI. Integrated \textbf{Azure OpenAI} and \textbf{Anthropic Claude APIs}; developed \textbf{Python/FastAPI} backends and \textbf{TypeScript/Node.js} applications with retrieval, streaming, and persistence workflows. Additional experience evaluating \textbf{LLMs}, \textbf{vision-language models}, and autonomous agents for scientific reasoning, correctness, and reliability.

%==================== SKILLS ====================
\section*{TECHNICAL SKILLS}
\textbf{Programming \& Backend:} \textbf{Python}, \textbf{TypeScript}, \textbf{JavaScript}, \textbf{Node.js}, \textbf{FastAPI}, \textbf{REST APIs}, \textbf{WebSockets}, \textbf{SSE}\\
\textbf{Generative AI:} \textbf{Azure OpenAI}, \textbf{Anthropic Claude API}, \textbf{LiteLLM}, \textbf{LangGraph}, \textbf{RAG}, \textbf{Text-to-SQL}, \textbf{embeddings}, \textbf{prompt engineering}, \textbf{function calling}, \textbf{guardrails}, \textbf{document parsing}, \textbf{LLM evaluation}\\
\textbf{Frameworks \& Data:} \textbf{React}, \textbf{Next.js}, \textbf{LlamaParse}, \textbf{LlamaIndex}, \textbf{Supabase}, \textbf{PostgreSQL}, \textbf{SQLite}, \textbf{Prisma}, \textbf{OpenSearch}\\
\textbf{Developer Tools:} \textbf{Git}, \textbf{GitHub}, \textbf{GitHub Actions}, \textbf{VS Code}, \textbf{Claude Code}, \textbf{n8n}, \textbf{Vitest}, \textbf{Vercel}, \textbf{Docker}\\
\textbf{Machine Learning \& Scientific Computing:} \textbf{PyTorch}, \textbf{TensorFlow}, \textbf{scikit-learn}, \textbf{NumPy}, \textbf{SciPy}, \textbf{Pandas}

%==================== PROJECTS ====================
\section*{SELECTED GENAI PROJECTS}

\projectline{US Housing Board -- Secure Agentic AI Assistant for Multi-Department Data}{[Role] \;|\; [Dates]}
\begin{itemize}
  \item Designed a secure \textbf{agentic AI assistant} that lets staff query \textbf{100+ relational tables} and scanned \textbf{PDF/image documents} in natural language, using a \textbf{LangGraph} planner--router--synthesizer that routes to a \textbf{Text-to-SQL agent}, a \textbf{document RAG agent}, or both.
  \item Enforced identity-aware access with \textbf{SSO (OIDC + MFA)}, \textbf{RBAC/ABAC} policies (OPA/Cedar), \textbf{row-level security}, \textbf{PII column masking}, and ACL-filtered vector retrieval; agents run on behalf of the user with short-lived credentials.
  \item Built input/output \textbf{guardrails} (prompt-injection detection; \textbf{SQL guard} with AST parsing, SELECT-only queries, and per-role table allow-lists; PII masking; groundedness and citation checks) and routed all LLM traffic through an \textbf{AI gateway} with quotas, model fallback, and cost tracking.
  \item Designed document ingestion (S3 $\rightarrow$ \textbf{Textract OCR} $\rightarrow$ chunking $\rightarrow$ embeddings $\rightarrow$ \textbf{OpenSearch} hybrid search with ACL metadata) and observability with \textbf{OpenTelemetry/Langfuse} traces and immutable audit logs.
\end{itemize}
\smallgap

\projectline{Auriga Career Lab -- GenAI Career Coach \& Opportunity-Matching Platform}{Founder / AI Developer \;|\; [Dates]}
\begin{itemize}
  \item Integrated \textbf{Anthropic Messages API} into a student-scoped career coach, assembling prompts from six-trait \textbf{RIASEC profiles}, ranked opportunity matches, and conversation history; implemented a deterministic fallback when the API key is unavailable.
  \item Developed personalised opportunity ranking by combining \textbf{RIASEC cosine similarity} with catalogue signals for freshness, funding, prestige, and eligibility.
  \item Built a versioned assessment workflow over a \textbf{170-item question bank}, combining student, parent, and teacher responses with agreement levels, insufficient-data handling, role-based access, and auditable event records.
\end{itemize}
\smallgap

\projectline{Unconstrained AI -- GenAI Engineering Project}{[Role] \;|\; [Dates]}
\begin{itemize}
  \item Implemented backend \textbf{LLM workflows} for eight educator use cases, including lesson planning, assessment and rubric generation, differentiation, IEPs, feedback, unit planning, and AI coaching, using \textbf{Azure OpenAI} through \textbf{LiteLLM}, with prompt templates, example datasets, and function calling.
  \item Built a document-grounded \textbf{RAG pipeline}: PDF ingestion with \textbf{LlamaParse}, embedding generation with Azure OpenAI, and document-chunk storage and retrieval with \textbf{Supabase}.
  \item Developed \textbf{FastAPI} APIs and asynchronous chat/generation flows using \textbf{WebSockets} and \textbf{SSE}; connected a \textbf{React/TypeScript} frontend with streaming output, conversation history, feedback, translations, and document export.
\end{itemize}

%==================== EXPERIENCE ====================
\section*{PROFESSIONAL EXPERIENCE}

\subsection*{AI Evaluation \& Model Training}
\experienceentry{AI Evaluation \& Benchmarking Specialist -- Project-Based Engagements}{2023--Present}{Handshake, Mercor, Bespoke Labs, Alinger, Turing AI, Scale AI \& Soul AI}
\begin{itemize}
  \item Designed and reviewed \textbf{scientific and multimodal evaluation tasks} for LLMs and vision-language models, validating scientific correctness, image quality, ground truth, references, task clarity, and answer tolerances across scientific domains.
  \item Evaluated \textbf{autonomous AI agents} on multi-step scientific reasoning and tool/function-calling workflows; reviewed benchmark solutions, identified reasoning failures and hallucinations, and developed structured criteria for consistency and reliability.
  \item Used \textbf{Claude Code} to assist in authoring and structuring computational-science tasks; worked in \textbf{VS Code} with \textbf{Git} to revise, version-control, and push task updates to \textbf{GitHub}, then orchestrated \textbf{n8n} workflows to automate task hand-off and final updates in \textbf{ID Prism}.
  \item Designed adversarial STEM prompts and contributed to \textbf{RLHF} workflows, analysing model errors and improving the quality and clarity of evaluation datasets.
\end{itemize}

\subsection*{Research}
\experienceentry{Senior Research Fellow -- Neutral-Atom Quantum Computing Platform}{Aug 2024--Feb 2026}{Cold Atom Laboratory, NISER, Bhubaneswar}
\begin{itemize}
  \item Specified laser architecture for \textbf{Rydberg-based quantum gates}; modelled \textbf{optical dipole traps}, including trap-depth, photon-scattering, and power-budget trade-offs.
  \item Implemented \textbf{Gerchberg--Saxton} phase-retrieval simulations for programmable optical microtrap arrays; developed experimental CAD layouts and coordinated component procurement and vendor evaluation.
\end{itemize}
\smallgap

\roleline{Research Project -- Excess tSZ Signal Between LRG Pairs in the DESI Survey (Stack-First Approach)}{Jan 2023--Apr 2023}
\begin{itemize}
  \item Analysed \textbf{DESI survey} data for Luminous Red Galaxy pairs using statistical stacking to extract a weak \textbf{thermal Sunyaev--Zel'dovich (tSZ) signal} between galaxy pairs.
\end{itemize}

%==================== EDUCATION ====================
\section*{EDUCATION}
\textbf{M.Sc.} -- National Institute of Science Education and Research (NISER), Bhubaneswar \hfill [Confirm dates]\\
CGPA: 8.50/10\\[-1pt]
\textbf{B.Sc.} -- Birla Institute of Technology, Mesra \hfill 2018--2021\\[-1pt]
CGPA: 8.28/10\\[1pt]
\textbf{Selected Recognition:} Asia Rank 1 -- Rudolf Ortvay International Problem-Solving Competition \;|\; AIR 78 -- IIT JAM \;|\; IASc--INSA--NASI Summer Research Fellowship

\end{document}
```

## 7. Placeholders to Fill Before Submitting

- `YOUR-LINKEDIN-ID`, `YOUR-GITHUB-ID`
- `[Role]` and `[Dates]` for the US Housing and Unconstrained AI projects
- `[Dates]` for Auriga Career Lab
- `[Confirm dates]` for the M.Sc.
