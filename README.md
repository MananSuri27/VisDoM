# 🧐📄 VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal RAG 🎯🤖  

Files for the NAACL 2025 paper, [**VisDoM**: Multi-Document QA with Visually Rich Elements Using **Multimodal Retrieval-Augmented Generation**](https://arxiv.org/abs/2412.10704). 📚📊🔍  


## 📂 What's Inside?  
This repo contains the **5 data splits** in **VisDoMBench**, a cutting-edge **multi-document, multimodal QA benchmark** 🚀🔎 designed for answering questions across **visually rich** document content like:  

📊 **Tables** | 📉 **Charts** | 🖼️ **Slides**  

Perfect for **evaluating** multimodal, multi-document QA systems in a **comprehensive** way! ✅📖  

![VisDoM Benchmark](https://github.com/user-attachments/assets/0d4f2220-90b4-41d0-b7f4-abd10471244b)  


## 📊 Dataset Summary  

| **Dataset**     | **Domain**             | **Content Type**             | **Queries** | **Docs** | **Avg. Question Length** | **Avg. Doc Length (Pages)** | **Avg. Docs per Query** | **Avg. Pages per Query** |
|----------------|-----------------------|-----------------------------|------------|---------|----------------------|----------------------|------------------|------------------|
| **PaperTab**   | Wikipedia             | Tables, Text                | 377        | 297     | 29.44 ± 6.3          | 10.55 ± 6.3          | 10.82 ± 4.4       | 113.10 ± 50.4    |
| **FetaTab**    | Scientific Papers     | Tables                      | 350        | 300     | 12.96 ± 4.1          | 15.77 ± 23.9         | 7.77 ± 3.1        | 124.33 ± 83.0    |
| **SciGraphQA** | Scientific Papers     | Charts                      | 407        | 319     | 18.05 ± 1.9          | 22.75 ± 29.1         | 5.91 ± 2.0        | 129.71 ± 81.7    |
| **SPIQA**      | Scientific Papers     | Tables, Charts              | 586        | 117     | 16.06 ± 6.6          | 14.03 ± 7.9          | 9.51 ± 3.5        | 135.58 ± 55.2    |
| **SlideVQA**   | Presentation Decks    | Slides                      | 551        | 244     | 22.39 ± 7.8          | 20.00 ± 0.0          | 6.99 ± 2.0        | 139.71 ± 40.6    |
| **VisDoMBench** | Combined              | Tables, Charts, Slides, Text | 2271       | 1277    | 19.11 ± 5.4          | 16.43 ± 14.5         | 8.36 ± 3.0        | 128.69 ± 62.7    |

📌 *Table: Summary of data splits included in VisDoMBench.*  

---  



## 📖 Cite us:  
```bibtex
@misc{suri2024visdommultidocumentqavisually,
      title={VisDoM: Multi-Document QA with Visually Rich Elements Using Multimodal Retrieval-Augmented Generation}, 
      author={Manan Suri and Puneet Mathur and Franck Dernoncourt and Kanika Goswami and Ryan A. Rossi and Dinesh Manocha},
      year={2024},
      eprint={2412.10704},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.10704}, 
}
```

