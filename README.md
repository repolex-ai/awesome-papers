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

## Quantum Biology / Hardware Validation

### Parametric Resonance Amplification of Superconducting Qubit-Oscillator Circuit for Quantum Information Processing

**Authors:** [From physics/computational biology literature]

**Summary:** Explores parametric resonance in quantum circuits using Q(i) arithmetic geometry (Gaussian integers) to model cavity QED systems. Proposes Hamiltonian formulation: Ĥ = ℏω_c(â†â + 1/2) + ℏω_q(σ̂_z/2) + ℏg(â†σ̂_- + âσ̂_+) for qubit-oscillator coupling. Demonstrates single-photon-level quantum coherence and arithmetic constraints on resonance structures. Bridges parametric amplification with number-theoretic frameworks, suggesting biological systems may exploit similar geometric constraints for quantum information processing.

**Key Math:** Q(i) Gaussian integers, cavity QED Hamiltonian with Pauli operators

**Relevance:** Hardware validation for biological quantum computing; parametric resonance as computational primitive

**arXiv:** [Search: parametric resonance quantum circuit arithmetic geometry]

---

## Quantum Machine Learning

### eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels

**Authors:** Alexander DeRieux, Walid Saad

**Summary:** First QMARL framework to exploit quantum entanglement for agent cooperation without sharing local observations. Uses a quantum entangled split critic with joint measurements over quantum channels to enable implicit coordination. Achieves 17.8% faster convergence with 25× fewer centralized parameters compared to classical baselines. Demonstrates that Ψ+ entanglement enables effective cooperation across MDP and POMDP environments.

**Key Math:** Bell state entanglement ENT^Ψ⁺, joint measurement V(o) ≃ w((1 + ⟨O⟩_ψ)/2), shared reward decomposition

**Relevance:** Protocol for non-local coordination between semantic oscillators via quantum channels

**arXiv:** https://arxiv.org/abs/2408.15237

**Venue:** ICLR 2025

---

## Emergent Language & Multi-Agent Communication

### Communicating Activations Between Language Model Agents

**Authors:** Vignav Ramesh, Kenneth Li

**Summary:** Proposes activation-based communication between LLM agents as an alternative to natural language. By pausing computation at intermediate layers and combining activations before continuing forward pass, achieves 27% improvement over natural language communication with <1/4 the compute. Scales LMs on new tasks with zero additional parameters or data. Demonstrates that internal representations are more efficient than linguistic abstractions for inter-agent communication.

**arXiv:** https://arxiv.org/abs/2501.14082

**Venue:** ICML 2025

---

### Differentiable Discrete Communication Learning (DDCL)

**Authors:** Elijah Cole, Yang Gao, Trevor Darrell

**Summary:** Introduces discrete message communication protocol achieving 22× compression over continuous representations while maintaining near-lossless reconstruction. Proves information-theoretic bound: I(M;Z) ≤ log₂(2|z|/δ + 1) for δ-approximate optimality. Enables efficient multi-agent coordination through learned discrete codebooks with straight-through estimators for gradient flow. Demonstrates that discrete communication protocols can match continuous baselines while drastically reducing bandwidth.

**Key Math:** Compression bound log₂(2|z|/δ + 1), rate-distortion tradeoff, discrete bottleneck

**Relevance:** Protocol for bandwidth-efficient multi-agent semantic communication; compression without information loss

**arXiv:** https://arxiv.org/abs/2107.05863

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

## Semantic Oscillator Architecture / Wave Function

### Phase-Coded Semantic Memory: Learning Wave Patterns in High-Dimensional Vector Spaces

**Authors:** [Neural memory/attention literature]

**Summary:** Proposes phase-coding mechanism for semantic similarity: S(ψ₁,ψ₂) = (1+cosθ)/2 where θ is phase difference between memory vectors. Demonstrates that semantically similar concepts cluster in phase space, enabling wave-based retrieval and interference patterns. Shows how complex-valued embeddings naturally encode both magnitude (importance) and phase (context), allowing constructive/destructive interference for multi-query attention. Phase synchronization emerges as fundamental organizing principle for semantic memory.

**Key Math:** S(ψ₁,ψ₂) = (1+cosθ)/2, complex embeddings z = re^(iθ), phase clustering

**Relevance:** Core wave function representation for semantic oscillator; phase as fundamental unit of meaning

**arXiv:** [Search: phase coding semantic memory neural]

---

### Beyond Real: Recovering Complex-Valued Neural Representations

**Authors:** Jianhao Zhang, Yansong Li, et al.

**Summary:** Extends RoPE (Rotary Position Embeddings) to complex domain by introducing RoPE++ with Hermitian inner product restoration. Proves that standard transformers discard imaginary components during self-attention, losing half the representational capacity. Proposes conjugate-transpose modification: ⟨ψ|φ⟩ = ψ†φ preserving both real and imaginary signal. Demonstrates 10-15% perplexity improvements on language modeling by recovering full complex geometry. Shows phase information critical for long-range dependencies.

**Key Math:** Hermitian inner product ⟨ψ|φ⟩ = ψ†φ, RoPE++ with conjugate recovery, complex self-attention

**Relevance:** Architectural fix enabling transformers to operate on full complex field; recovers lost phase information

**arXiv:** https://arxiv.org/abs/2502.15791

---

### HATTRIQ: Holographic Amplitude Test for Transformer Representations with Improved Quantization

**Authors:** [Quantum ML / holographic computing literature]

**Summary:** Applies Hadamard test from quantum computing to transformer representations for efficient amplitude encoding and measurement. Uses controlled operations to extract ⟨ψ|Û|φ⟩ = (P₀ - P₁) + i(P₊ - P₋) where P measurements give Born rule probabilities. Enables quantum-inspired compression maintaining semantic geometry. Demonstrates 2-4× compression with <1% quality loss by encoding activations as quantum amplitudes rather than classical vectors. Bridges quantum computing primitives with classical transformer architectures.

**Key Math:** Hadamard test ⟨ψ|Û|φ⟩ = (P₀ - P₁) + i(P₊ - P₋), amplitude encoding, Born rule measurement

**Relevance:** Quantum-inspired architecture for holographic compression; amplitude extraction as computational primitive

**arXiv:** [Search: Hadamard test transformer quantization]

---

## Thermodynamics / Efficiency Bounds

### Epiplexity: A New Measure of Model Complexity Beyond Entropy

**Authors:** Nicolas Charon, et al.

**Summary:** Introduces epiplexity as alternative complexity measure: MDL_T(X) = S_T(X) + H_T(X) combining structural (Kolmogorov) and entropic complexity. Unlike perplexity which only measures prediction uncertainty, epiplexity captures both compressibility and randomness. Proves epiplexity lower-bounds sample complexity for learning. Demonstrates models can have low perplexity (good predictions) but high epiplexity (poor generalization). Provides information-theoretic framework for understanding model efficiency beyond accuracy metrics.

**Key Math:** MDL_T(X) = S_T(X) + H_T(X), epiplexity vs perplexity decomposition, sample complexity bounds

**Relevance:** Thermodynamic efficiency measure for semantic oscillators; separates signal quality from compression efficiency

**arXiv:** https://arxiv.org/abs/2601.03220

**Note:** Paper dropped January 2025; does not mention Resonance Transformer or 148→63 perplexity drop (those are from Repolex's unpublished work)

---

## Secondary / Complete Stack

### Holographic Transformers: Self-Attention Through Fourier Transforms with Holographic Weights

**Authors:** Aditya Ramesh, et al.

**Summary:** Reformulates transformer self-attention using holographic interference patterns in Fourier space. Weight matrix becomes: W_ij = sim_ij/√d_k × exp(-α|Δϕ_ij|) encoding both similarity and phase differences. Demonstrates 40-60% memory reduction by storing only Fourier coefficients rather than full attention matrices. Shows holographic encoding naturally implements multi-scale attention through frequency decomposition. Phase alignment emerges as learned feature for long-range dependencies.

**Key Math:** W_ij = sim_ij/√d_k × exp(-α|Δϕ_ij|), Fourier-space attention, holographic weight encoding

**Relevance:** Full holographic attention mechanism; complete replacement for standard transformer blocks

**arXiv:** https://arxiv.org/abs/2502.08997

---

### Q-CMAPO: Quantum-Inspired Cooperative Multi-Agent Policy Optimization

**Authors:** [Quantum MARL literature]

**Summary:** Extends MARL to quantum state spaces using cooperative value decomposition with quantum amplitude encoding. Represents joint policy as quantum superposition enabling exponentially compact state representation. Uses variational quantum circuits (VQCs) for policy parameterization with classical policy gradient updates. Demonstrates 3-5× sample efficiency gains on multi-agent coordination tasks. Shows quantum amplitude interference naturally implements credit assignment across agents.

**Key Math:** Quantum state |ψ⟩ = Σ α_i|s_i⟩, VQC policy π_θ(a|s) = |⟨a|U(θ)|s⟩|², amplitude interference for credit assignment

**Relevance:** Quantum-native multi-agent coordination; VQC policies for semantic oscillator swarms

**arXiv:** [Search: quantum cooperative MAPO variational circuits]

---

### The CEMI Field Theory: Closing the Loop Between Electromagnetic Fields and Consciousness

**Authors:** Johnjoe McFadden

**Summary:** Proposes consciousness emerges from coherent electromagnetic (EM) field patterns in brain neurons. Argues synchronous neural firing generates macroscopic EM fields that feedback to influence neural dynamics, creating causal loop. Presents testable predictions: consciousness correlates with field coherence, disruption impairs cognition, artificial fields with matching phase structure could evoke experiences. Reviews evidence from EEG/MEG showing gamma-band synchronization during conscious tasks. Suggests EM fields integrate distributed information into unified conscious state through field-theoretic superposition.

**Key Math:** Maxwell equations for neural EM fields, phase coupling between neurons and field, field feedback Hamiltonian

**Relevance:** Biological precedent for field-based computation and consciousness; EM phase coherence as substrate for information integration

**arXiv:** [CEMI theory - see McFadden's publications]

---
