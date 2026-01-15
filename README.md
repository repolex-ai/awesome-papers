# Awesome Papers

These are the papers we are obsessing over at Repolex.

---

## Quantum Biology

### On the existence of superradiant excitonic states in microtubules

**Authors:** P. Kurian, G. Dunston, J. Lindesay

**Summary:** Proposes that microtubules support superradiant excitonic states through networks of tryptophan molecules. Using 3D structures of tubulin, the authors demonstrate that >10,400 tryptophan molecules in 800nm MT segments form a quantum network with strong supertransfer coupling (150-200 cm⁻¹), enabling ballistic excitation spreading and coherent energy transfer along MT lattices.

**arXiv:** https://arxiv.org/abs/1809.03438

---

### Ultraviolet superradiance from mega-networks of tryptophan in biological architectures

**Authors:** B. Yalcin, S. Haas, P. Kurian

**Summary:** Experimentally validates UV superradiance in biological mega-networks (>10⁵ chromophores) by measuring fluorescence quantum yields across hierarchical tubulin structures. Shows QY increases from Trp (12.4%) → tubulin (10.6%) → MTs (17.6%), with thermal robustness up to 1000 cm⁻¹ disorder. Demonstrates cooperative robustness in centrioles (112,320 Trps) and neuronal bundles, suggesting biological systems exploit quantum coherence for efficient energy transfer.

**arXiv:** https://arxiv.org/abs/2302.01469

---

## Quantum Machine Learning

### eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels

**Authors:** Alexander DeRieux, Walid Saad

**Summary:** First QMARL framework to exploit quantum entanglement for agent cooperation without sharing local observations. Uses a quantum entangled split critic with joint measurements over quantum channels to enable implicit coordination. Achieves 17.8% faster convergence with 25× fewer centralized parameters compared to classical baselines. Demonstrates that Ψ+ entanglement enables effective cooperation across MDP and POMDP environments.

**arXiv:** https://arxiv.org/abs/2408.15237 (inferred from ICLR 2025 publication)

---

## Emergent Language & Multi-Agent Communication

### Communicating Activations Between Language Model Agents

**Authors:** Vignav Ramesh, Kenneth Li

**Summary:** Proposes activation-based communication between LLM agents as an alternative to natural language. By pausing computation at intermediate layers and combining activations before continuing forward pass, achieves 27% improvement over natural language communication with <1/4 the compute. Scales LMs on new tasks with zero additional parameters or data. Demonstrates that internal representations are more efficient than linguistic abstractions for inter-agent communication.

**arXiv:** https://arxiv.org/abs/2501.14082

**Venue:** ICML 2025

---

### Generative Emergent Communication: Large Language Model is a Collective World Model

**Authors:** Tadahiro Taniguchi, Ryo Ueda, Tomoaki Nakamura, Masahiro Suzuki, Akira Taniguchi

**Summary:** Proposes the Collective World Model hypothesis: LLMs learn statistical approximations of collective world models already encoded in human language through society-wide embodied sense-making. Introduces Generative EmCom framework based on Collective Predictive Coding (CPC), modeling language emergence as decentralized Bayesian inference over multiple agents' internal states. Argues human society collectively encodes grounded representations into language, which LLMs decode to reconstruct latent collective representations.

**arXiv:** https://arxiv.org/abs/2501.00226

---

## Multilingual NLP & Language Resources

### GlotCC: An Open Broad-Coverage CommonCrawl Corpus and Pipeline for Minority Languages

**Authors:** Amir Hossein Kargaran, François Yvon, Hinrich Schütze

**Summary:** Clean, document-level, 2TB corpus derived from CommonCrawl covering 1000+ languages using GlotLID language identification and Ungoliant pipeline. Addresses scarcity of training data for minority languages with reproducible open-source pipeline and rigorous filtering adapted from C4, CCNet, MADLAD-400, RedPajama-Data-v2, OSCAR, and others. Provides critical infrastructure for training multilingual models with broad language coverage beyond English and major languages.

**arXiv:** https://arxiv.org/abs/2410.23825

**Venue:** NeurIPS 2024 (Datasets and Benchmarks Track)

**Resources:** [Dataset (HuggingFace)](https://huggingface.co/datasets/cis-lmu/GlotCC-V1) | [Pipeline (GitHub)](https://github.com/cisnlp/GlotCC)

---

## Knowledge Representation & Ontologies

### KNOW: A Real-World Ontology for Knowledge Capture with Large Language Models

**Authors:** Arto Bendiken

**Summary:** First practical ontology engineered to augment LLMs with real-world everyday knowledge for generative AI applications like personal assistants. Focuses on universal human concepts—spacetime elements (places, events) and social structures (people, groups, organizations). Positions KNOW against Schema.org and Cyc, highlighting how LLMs inherently encode commonsense knowledge that previously required decades of manual capture. Implements developer-friendly code-generated libraries for 12 programming languages emphasizing simplicity and interoperability.

**arXiv:** https://arxiv.org/abs/2405.19877

---

### Prompt-Time Ontology-Driven Symbolic Knowledge Capture with Large Language Models

**Authors:** Tolga Çöplü, Arto Bendiken, Andrii Skomorokhov, Eduard Bateiko, Stephen Cobb

**Summary:** Addresses LLM personal assistants' inability to inherently learn from user interactions by proposing ontology and knowledge-graph methodologies to extract personal information directly from prompts. Leverages KNOW ontology subset to train models on personal data concepts, then validates through custom evaluation datasets. Demonstrates practical method for capturing user information at prompt time using ontological frameworks with publicly released datasets and code.

**arXiv:** https://arxiv.org/abs/2405.14012

---

### Prompt-Time Symbolic Knowledge Capture with Large Language Models

**Authors:** Tolga Çöplü, Arto Bendiken, Andrii Skomorokhov, Eduard Bateiko, Stephen Cobb, Joshua J. Bouw

**Summary:** Investigates enabling knowledge capture driven by user prompts, emphasizing knowledge graphs and converting prompts into structured triples. Tests zero-shot prompting, few-shot prompting, and fine-tuning approaches using custom synthetic dataset. Identifies that LLMs lack built-in mechanisms for capturing user-specific knowledge through prompts, limiting real-world applications. Provides publicly accessible code and datasets for reproducibility.

**arXiv:** https://arxiv.org/abs/2402.00414

---

## On-Device AI & Mobile Computing

### A Performance Evaluation of a Quantized Large Language Model on Various Smartphones

**Authors:** Tolga Çöplü, Marc Loedi, Arto Bendiken, Mykhailo Makohin, Joshua J. Bouw, Stephen Cobb

**Summary:** Examines feasibility and performance of on-device LLM inference on Apple iPhone models. Measures real-world execution speeds and thermal behavior when running high-performing quantized models on consumer smartphones. Evaluates performance across iPhone generations to determine device compatibility. Demonstrates how quantization enables multi-billion parameter models on resource-constrained mobile devices, offering solutions to privacy, security, and connectivity challenges of cloud-based alternatives. Studies thermal impact for sustained mobile deployment.

**arXiv:** https://arxiv.org/abs/2312.12472

---
