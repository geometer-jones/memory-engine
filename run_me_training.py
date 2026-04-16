"""Fine-tune Memory Engine layers on WikiText-2 while keeping GPT-2 frozen.

The decisive test: can the framework's operations (Hadamard reception,
regime-aware update, renormalization, recurrence) be trained to improve
the model's next-token prediction?
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from me_layer import create_model
from llm_instrument import compute_pr, compute_anisotropy

np.set_printoptions(precision=4, suppress=True)


class TextDataset(Dataset):
    """Simple tokenized text dataset."""

    def __init__(self, texts: list[str], tokenizer, block_size: int = 128):
        self.examples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            # Use overlapping windows with stride = block_size // 2
            stride = max(1, block_size // 2)
            for i in range(0, len(tokens) - block_size, stride):
                self.examples.append(tokens[i:i + block_size + 1])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens = self.examples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


def make_wikitext_data(tokenizer, block_size=256, max_docs=200):
    """Load a small slice of WikiText-2 (no HF datasets dependency)."""
    # Use a simple fallback: generate training data from diverse text
    texts = [
        "The history of computing spans thousands of years, from ancient abacuses to modern supercomputers. "
        "Early mechanical calculators paved the way for electronic computers in the twentieth century. "
        "The invention of the transistor revolutionized computing, enabling smaller and faster machines. "
        "Programming languages evolved from machine code to high-level languages like Fortran and COBOL. "
        "The development of the internet connected computers worldwide, transforming communication and commerce. "
        "Modern artificial intelligence systems can process natural language, recognize images, and generate code. "
        "Quantum computing promises to solve problems that are intractable for classical computers. "
        "Cloud computing allows organizations to scale their infrastructure dynamically without large capital investments. "
        "Cybersecurity has become a critical concern as more business operations move online. "
        "Open source software has accelerated innovation by enabling collaborative development across organizations.",

        "Neural networks are computational systems inspired by biological brains. They consist of layers of "
        "interconnected nodes that process information. Deep learning uses many layers to extract hierarchical "
        "features from raw data. Convolutional networks excel at image recognition tasks. Recurrent networks "
        "process sequential data such as text and time series. The transformer architecture introduced attention "
        "mechanisms that revolutionized natural language processing. Large language models demonstrate emergent "
        "capabilities including reasoning, translation, and code generation. Training these models requires "
        "massive computational resources and carefully curated datasets. Transfer learning allows pretrained "
        "models to be adapted to specific tasks with minimal additional training. Reinforcement learning from "
        "human feedback has improved the alignment of language models with human preferences.",

        "The solar system consists of the Sun and everything that orbits it. Eight planets circle the Sun "
        "in roughly the same plane. Mercury and Venus are rocky planets closest to the Sun. Earth is the "
        "third planet and the only known world to support life. Mars has a thin atmosphere and evidence of "
        "ancient water. Jupiter is the largest planet, a gas giant with dozens of moons. Saturn is known "
        "for its spectacular ring system. Uranus and Neptune are ice giants in the outer solar system. "
        "Beyond Neptune lies the Kuiper Belt, home to Pluto and many other icy bodies. The Oort Cloud "
        "marks the distant boundary of the solar system, containing trillions of comets.",

        "Evolution by natural selection is the process by which organisms change over generations. Charles "
        "Darwin and Alfred Russel Wallace independently developed the theory in the nineteenth century. "
        "Genetic variation arises through mutations in DNA. Organisms with advantageous traits are more "
        "likely to survive and reproduce. Over millions of years, this process produces new species. The "
        "fossil record provides evidence for evolutionary transitions. DNA sequencing has revealed the "
        "molecular mechanisms underlying inheritance. Comparative genomics shows how closely related "
        "different species are. Evolutionary principles are applied in medicine, agriculture, and "
        "conservation biology. The study of evolution continues to yield insights into the diversity of life.",

        "Mathematics is the study of abstract structures, patterns, and relationships. Number theory "
        "examines the properties of integers and prime numbers. Algebra generalizes arithmetic operations "
        "to abstract structures like groups, rings, and fields. Geometry studies shapes, spaces, and their "
        "transformations. Calculus provides tools for analyzing change and accumulation. Statistics and "
        "probability quantify uncertainty and variation. Topology studies properties preserved under "
        "continuous deformations. Mathematical proofs establish truths through logical deduction from axioms. "
        "Applied mathematics uses these tools to model physical, biological, and social systems. The boundary "
        "between pure and applied mathematics continues to shift as new applications emerge.",

        "The human brain contains approximately 86 billion neurons connected by trillions of synapses. "
        "Neurons communicate through electrical impulses and chemical neurotransmitters. The cerebral cortex "
        "is responsible for higher cognitive functions including language, planning, and abstract thought. "
        "The hippocampus plays a crucial role in forming new memories. The cerebellum coordinates movement "
        "and balance. Neuroplasticity allows the brain to reorganize itself in response to experience. "
        "Sleep is essential for memory consolidation and cognitive function. Emotions involve complex "
        "interactions between the amygdala, prefrontal cortex, and autonomic nervous system. Neurotransmitter "
        "imbalances are associated with conditions like depression and Parkinson's disease. Advances in "
        "neuroimaging have revealed the brain's remarkable complexity and adaptability.",

        "Climate change refers to long-term shifts in global temperatures and weather patterns. Human "
        "activities, particularly the burning of fossil fuels, have increased atmospheric carbon dioxide "
        "concentrations significantly since the industrial revolution. Rising temperatures cause glaciers "
        "to melt, sea levels to rise, and weather patterns to shift. Extreme weather events including "
        "heatwaves, floods, and hurricanes are becoming more frequent and severe. Ecosystems face pressure "
        "as species struggle to adapt to rapidly changing conditions. Renewable energy sources like solar, "
        "wind, and hydropower offer alternatives to fossil fuels. Carbon capture technology aims to remove "
        "carbon dioxide directly from the atmosphere. International agreements like the Paris Accord seek "
        "to coordinate global action on emissions reduction. Adaptation strategies include improving "
        "infrastructure resilience and developing drought-resistant crops.",

        "Philosophy examines fundamental questions about existence, knowledge, values, reason, and language. "
        "Epistemology investigates the nature and limits of human knowledge. Metaphysics explores questions "
        "about the fundamental nature of reality. Ethics examines concepts of right and wrong, good and bad. "
        "Logic provides tools for valid reasoning and argumentation. The philosophy of mind investigates "
        "the nature of consciousness and its relationship to the physical brain. Political philosophy "
        "examines questions of justice, authority, and the ideal state. Aesthetics studies the nature of "
        "beauty and art. The history of philosophy spans traditions from ancient Greece to modern analytic "
        "and continental approaches. Philosophy continues to inform and be informed by scientific discoveries.",

        "Architecture is the art and science of designing and constructing buildings and other structures. "
        "Ancient civilizations developed distinctive architectural styles reflecting their cultures and "
        "technologies. Greek architecture introduced the classical orders: Doric, Ionic, and Corinthian. "
        "Roman engineering produced aqueducts, amphitheaters, and domed structures using concrete. Gothic "
        "cathedrals featured pointed arches, ribbed vaults, and flying buttresses. The Renaissance revived "
        "classical proportions and symmetry. Modern architecture embraced new materials like steel and glass, "
        "prioritizing function over ornament. Sustainable architecture seeks to minimize environmental impact "
        "through energy efficiency and renewable materials. Digital design tools have expanded the range of "
        "geometrically complex forms that can be realized. Architecture shapes human experience by defining "
        "the spaces in which we live, work, and gather.",

        "Music is a universal form of human expression that combines rhythm, melody, and harmony. "
        "Different cultures have developed distinct musical traditions spanning thousands of years. "
        "Western classical music evolved from medieval chant through Baroque, Classical, Romantic, and "
        "modern periods. Jazz originated in the early twentieth century, blending African rhythmic "
        "traditions with European harmony. Rock and roll emerged in the 1950s, drawing from blues, "
        "country, and gospel music. Electronic music uses synthesizers and digital technology to create "
        "new sonic possibilities. Music theory provides a framework for understanding melody, harmony, "
        "rhythm, and form. The psychology of music explores how humans perceive, create, and respond to "
        "musical sounds. Music therapy uses musical activities to address physical, emotional, and "
        "cognitive needs. Recording technology has transformed how music is produced and distributed.",
    ]

    return texts


def evaluate_perplexity(model, tokenizer, texts, block_size=256, device="cpu"):
    """Compute perplexity on text data."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - block_size, block_size):
                input_ids = torch.tensor([tokens[i:i + block_size]]).to(device)
                labels = input_ids.clone()
                outputs = model(input_ids, labels=labels)
                if outputs["loss"] is not None:
                    total_loss += outputs["loss"].item() * block_size
                    total_tokens += block_size

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def train_me_layers(
    n_epochs=5,
    block_size=128,
    lr=0.01,
    device="cpu",
):
    """Train Memory Engine layers while keeping GPT-2 frozen."""

    print("Loading model...")
    model, tokenizer = create_model("gpt2", insert_after=[3, 6, 9])
    model.to(device)

    # Training data
    texts = make_wikitext_data(tokenizer, block_size)
    # Split into train/eval
    n_eval = 2
    eval_texts = texts[:n_eval]
    train_texts = texts[n_eval:]

    # Debug: check dataset sizes
    train_ds = TextDataset(train_texts, tokenizer, block_size)
    eval_ds = TextDataset(eval_texts, tokenizer, block_size)
    print(f"  Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    if len(train_ds) == 0:
        # Concatenate all texts into longer sequences
        print("  Texts too short, concatenating...")
        train_texts = [" ".join(train_texts)]
        eval_texts = [" ".join(eval_texts)]

    train_dataset = TextDataset(train_texts, tokenizer, block_size)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Only train ME parameters
    optimizer = torch.optim.Adam(model.get_me_parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Baseline perplexity
    print("\nComputing baseline perplexity...")
    baseline_ppl = evaluate_perplexity(model, tokenizer, eval_texts, block_size, device)
    print(f"Baseline perplexity: {baseline_ppl:.2f}")

    # Log ME parameter state before training
    print(f"\nME parameters before training:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")

    # Training loop
    print(f"\nTraining for {n_epochs} epochs...")
    print(f"{'Epoch':>5} {'Loss':>8} {'LR':>8} {'PPL':>8}")
    print("-" * 35)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.get_me_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate
        avg_loss = epoch_loss / max(n_batches, 1)
        ppl = evaluate_perplexity(model, tokenizer, eval_texts, block_size, device)

        print(f"  {epoch+1:>3} {avg_loss:>8.3f} {scheduler.get_last_lr()[0]:>8.5f} {ppl:>8.2f}")

    # Final state
    print(f"\nME parameters after training:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")

    # Diagnostics: PR at final layer before and after
    print(f"\nFinal comparison:")
    test_text = eval_texts[0]
    input_ids = tokenizer.encode(test_text[:200], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        hs = outputs["hidden_states"]

    final_pr = np.mean([compute_pr(hs[-1].squeeze(0)[pos].cpu().numpy())
                        for pos in range(hs[-1].shape[1])])
    print(f"  Final layer PR: {final_pr:.1f}")
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    delta_ppl = baseline_ppl - ppl
    if delta_ppl > 0:
        print(f"  PPL improvement: {delta_ppl:.2f} (lower is better)")
    else:
        print(f"  PPL regression: {delta_ppl:.2f}")

    return model, {
        "baseline_ppl": baseline_ppl,
        "final_ppl": ppl,
        "final_pr": final_pr,
    }


if __name__ == "__main__":
    device = "cpu"
    model, results = train_me_layers(
        n_epochs=5,
        block_size=256,
        lr=0.01,
        device=device,
    )

    print(f"\n{'=' * 60}")
    print(f"Results: Baseline PPL={results['baseline_ppl']:.2f}, "
          f"Final PPL={results['final_ppl']:.2f}, "
          f"PR={results['final_pr']:.1f}")
    print(f"{'=' * 60}")
