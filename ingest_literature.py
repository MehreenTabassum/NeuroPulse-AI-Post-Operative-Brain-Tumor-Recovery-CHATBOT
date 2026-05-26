"""
ingest_literature.py — Populate the Chroma vector store with BraTS-2024 literature.

Run ONCE before starting the stack:
    python ingest_literature.py

Requires Ollama running with the nomic-embed-text model:
    ollama pull nomic-embed-text
"""

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


def main() -> None:
    persist_directory = "./chroma_db"

    print("1. Connecting to Ollama for embeddings...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    print("2. Preparing clinical literature & BraTS-2024 guidelines...")
    sample_docs = [
        Document(
            page_content=(
                "Clinical guidelines for Glioblastoma Multiforme (GBM) post-operative monitoring. "
                "Following gross total resection (GTR), patients undergoing the Stupp protocol "
                "(Temozolomide + Radiotherapy) should be evaluated at 12 weeks post-surgery using RANO "
                "criteria to carefully differentiate true tumor progression from treatment-induced "
                "radiation necrosis."
            ),
            metadata={"source": "Neuro-Oncology Review 2025", "type": "guideline"},
        ),
        Document(
            page_content=(
                "Analysis of the UCSD-PTGBM-BraTS-2024 dataset indicates that 768-dimensional deep "
                "feature vectors extracted from co-registered T1, T1CE, T2, and FLAIR MRI sub-regions "
                "act as strong predictors for long-term physiological recovery and Karnofsky Performance "
                "Scale (KPS) stability."
            ),
            metadata={"source": "BraTS 2024 Proceedings", "type": "dataset_spec"},
        ),
        Document(
            page_content=(
                "Radiation necrosis vs Tumor Recurrence: Radiation necrosis typically presents within "
                "3 to 6 months post-chemoradiation. It is characterized by circumscribed T1CE enhancement "
                "without true mass effect, coupled with stabilizing KPS scores (e.g., KPS around 80), "
                "whereas true progression features structural expansion and worsening neurological deficits."
            ),
            metadata={"source": "Lancet Neurology Abstract", "type": "differential_diagnosis"},
        ),
        Document(
            page_content=(
                "IDH-wildtype GBM patients have a median overall survival of 14–16 months with Stupp "
                "protocol. MGMT promoter methylation is a positive predictive biomarker for temozolomide "
                "response, with methylated patients showing improved progression-free survival of 10.3 "
                "versus 5.9 months in unmethylated cohorts."
            ),
            metadata={"source": "NEJM GBM Biomarkers 2024", "type": "biomarkers"},
        ),
        Document(
            page_content=(
                "KPS score trajectory is a key indicator of functional recovery post-GBM surgery. "
                "A KPS >= 70 at 12 weeks post-surgery is associated with eligibility for further "
                "adjuvant therapy. Decline below KPS 60 may indicate pseudoprogression or true "
                "tumor recurrence and warrants urgent MRI re-evaluation under RANO criteria."
            ),
            metadata={"source": "Journal of Neuro-Oncology 2024", "type": "functional_outcomes"},
        ),
        Document(
            page_content=(
                "BraTS 2024 challenge dataset contains multi-parametric MRI scans from post-operative "
                "GBM patients. The dataset includes T1, T1CE, T2, and FLAIR sequences with expert "
                "segmentation labels for enhancing tumor (ET), tumor core (TC), and whole tumor (WT) "
                "regions. Feature vectors extracted using CNN-ViT architectures achieve AUC 0.87 for "
                "progression prediction at 6-month follow-up."
            ),
            metadata={"source": "BraTS 2024 Dataset Card", "type": "dataset_spec"},
        ),
    ]

    print(f"3. Ingesting {len(sample_docs)} documents into Chroma at '{persist_directory}'...")

    # FIX: Chroma.from_documents handles persistence automatically in chromadb>=0.4.
    # The deprecated .persist() call has been removed.
    vectorstore = Chroma.from_documents(
        documents=sample_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="brats2024_literature",
    )

    # Verify ingestion worked
    results = vectorstore.similarity_search("GBM recovery KPS", k=2)
    print(f"4. Verification — sample query returned {len(results)} docs.")
    for r in results:
        print(f"   → {r.metadata.get('source')} | {r.page_content[:80]}...")

    print("\n✔ Vector store populated successfully!")
    print(f"  Location : {persist_directory}/")
    print("  Run 'python run_all.py' to start the full stack.\n")


if __name__ == "__main__":
    main()
