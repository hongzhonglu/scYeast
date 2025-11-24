scYeast: Deep Learning Framework for Yeast Transcriptome and Proteome Analysis
This project is a deep learning framework designed for the analysis of yeast transcriptome and proteome data. It includes pre-training based on transcriptome data, fine-tuning for downstream tasks (such as pressure response prediction), as well as pre-training and downstream tasks based on proteome data.

Project Structure
scYeast/
├── model_construction/        # Transcriptome pre-training model construction
├── models/                    # Pre-trained model files
├── pressure_finetune/         # Pressure response fine-tuning tasks
├── proteome_pretrain/         # Proteome pre-training
├── proteome_finetune/         # Proteome downstream task fine-tuning
├── fine_tune/                 # Other fine-tuning tasks
├── analysis/                  # Analysis scripts
├── zero_shot/                 # Zero-shot learning tasks
└── requirements.txt           # Python dependencies
Important Note: Handling Large Files
Due to GitHub's single file size limit of 2GB, ultra-large files in this project have been split.

Merging Split Files
After cloning the repository, you must merge the following split files before running the code:

1. Merge Protein Interaction Data (7.4GB)
Bash

cd proteome_finetune/data
bash merge_pro_interaction.sh
# You can delete the split files after merging to save space
# rm pro_interaction.pkl.part_*
2. Merge Alignment Data (2.4GB)
Bash

cd proteome_pretrain/data
bash merge_alignment_data.sh
# You can delete the split files after merging to save space
# rm alignment_data.txt.part_*
Note: The merging process requires sufficient disk space (at least 15GB free).

Installation
1. Create Conda Environment
Bash

conda create -n scyeast python=3.9
conda activate scyeast
2. Install Dependencies
Bash

# Install based on requirements.txt (conda list --export format)
conda install --file requirements.txt

# Or install main dependencies using pip
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn scipy
pip install matplotlib seaborn
pip install goatools  # For GO enrichment analysis
3. Install Git LFS (For Large File Management)
If you need to clone the full model and data files:

Bash

# Install Git LFS
git lfs install

# Clone the repository (LFS files will be downloaded automatically)
git clone https://github.com/hongzhonglu/scYeast.git
cd scYeast
Pre-trained Models
The project utilizes the following pre-trained models:

models/final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth - Transcriptome pre-trained model with knowledge graph integration.

models/final_checkpoint_spearman0.5_wo_knowledge_huber_v4.pth - Transcriptome pre-trained model without knowledge graph integration.

Usage Guide
1. Pressure Response Prediction (Pressure Fine-tuning)
The pressure response prediction task is used to predict the phenotypic response of yeast under different stress conditions.

Environment Preparation
Bash

cd pressure_finetune
Data Files
alignment_pressure_data_1_with_labels.csv - Pressure response dataset (features and labels)

GSE201387_yeast.tpm.tsv - TPM expression data

pressure_DEGs.xlsx - Differentially Expressed Genes

log(tpm+1)d2_pressure_data_1.csv - Pre-processed expression data

Running Fine-tuning
Fine-tuning with Pre-trained Model
Bash

python pressure_fine_tune.py
This script will:

Load data from alignment_pressure_data_1_with_labels.csv.

Load the pre-trained model ../models/final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth.

Run 10 independent experiments to evaluate model stability.

Save the best model from each run to pressure_fine_tune_best_model_run_{i}.pth.

Save prediction results to the results/ directory.

Training from Scratch (No Pre-training)
Bash

python pressure_no_pretrain.py
Used for comparative experiments to evaluate the effect of pre-training.

Pre-training without Knowledge Graph
Bash

python pressure_wo_know.py
Result Analysis
After training, result files are saved in the results/ directory:

pressure_test_predictions.npy - Test set predicted probabilities

pressure_test_true_labels.npy - Test set true labels

pressure_fine_tune_best_model_run_*.pth - Best models from each run

Visualize results using ROC curves:

Bash

python roc_paint.py
Other Machine Learning Baselines
The project provides various ML methods for comparison:

Bash

python svm.py           # Support Vector Machine
python knn.py           # K-Nearest Neighbors
python decisiontree.py  # Decision Tree
python naive_bayes.py   # Naive Bayes
2. Proteome Pre-training
Proteome pre-training is based on the transcriptome pre-trained model, further trained using proteome data.

Environment Preparation
Bash

cd proteome_pretrain
Data Files
Data files are located in the data/ directory:

alignment_data.txt - Aligned proteome data (2.4GB, via Git LFS)

scaled_data_1_pro.pkl - Scaled protein data (85MB, via Git LFS)

gene2vec_dim_200_iter_9spearman0.5.txt - Gene vector representations

filter_proindex.pkl - Filtered protein index

final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth - Transcriptome pre-trained model

Running Pre-training
Continue Training Based on Transcriptome Model
Bash

python "DSgraph_main_v4 _pro_test.py"
This script will:

Load the transcriptome pre-trained model.

Continue training using proteome data.

Save the trained model as final_checkpoint_spearman0.5_w_knowledge_v4pro.pth.

Output training logs to output_pro.txt.

Training from Scratch (No Transcriptome Pre-training)
Bash

python "DSgraph_main_v4 _pro_nopretrain.py"
Used for comparative experiments to evaluate the transfer learning effect of transcriptome pre-training.

Model Architecture
Proteome pre-training uses the following core modules:

DSgraph_pro.py - Main model architecture

embedding_pro.py - Embedding layer

self_attention_pro.py - Self-attention mechanism

gated_fusion.py - Gated fusion layer

gene2vec.py - Gene vectorization

load_data.py - Data loading utility

3. Proteome Downstream Task Fine-tuning
Fine-tuning downstream tasks based on the proteome pre-trained model, including Growth Rate prediction, Ribosome Profiling (RPF) prediction, and Protein Turnover prediction.

Environment Preparation
Bash

cd proteome_finetune
Data Files
Data files are located in the data/ directory (via Git LFS):

pro_interaction.pkl - Protein interaction data (7.4GB)

scaled_data_1_pro.pkl - Scaled protein data (85MB)

growth_tensor.pkl - Growth data

pretrained_features_max.pkl - Pre-trained features (Max)

pretrained_features_min.pkl - Pre-trained features (Min)

Other pkl files: Gene index, Sample index, Filters, etc.

Running Fine-tuning Tasks
Growth Rate Prediction
Bash

python growth_finetune.py
Ribosome Profiling (RPF) Prediction
Bash

python rpf_finetune.py
Uses K-fold cross-validation. Results saved in rpf_kfold_results/.

Protein Turnover Prediction
Bash

python turnover_finetune.py
Uses K-fold cross-validation. Results saved in turnover_kfold_results/.

Result Files
rpf_kfold_results/ - K-fold validation results for RPF

fold_*/model.pkl - Model for each fold

fold_*/predictions.csv - Predictions for each fold

round_results.csv - Summary results

turnover_kfold_results/ - K-fold validation results for Turnover (same structure)

ml_model_results/ - Machine learning model results

extra_tree_fold_*.pkl - Extra Trees model (169MB, via Git LFS)

fold_*_predictions.csv - Prediction results

Analysis Tools
Various analysis scripts are provided in the analysis/ directory:

GO Enrichment Analysis
Bash

cd analysis
python GO_weighted_enrich.py  # Weighted GO enrichment analysis
python GO_paint.py            # Visualization of GO enrichment results
Performance Evaluation
Bash

python r2_scatter_paint.py    # R² Scatter Plot
python loss_compare.py        # Loss function comparison
Zero-shot Learning
Located in the zero_shot/ directory:

gene_embedding_analysis.py - Gene embedding analysis

gene_pair_analysis.py - Gene pair analysis

GRN.py - Gene Regulatory Network analysis

Notes
Large File Management
This project uses Git LFS to manage large files (>50MB):

Model files (*.pth)

Data files (*.pkl, *.npy)

Large text files (alignment_data.txt)

Ensure Git LFS is installed:

Bash

git lfs install
GPU Requirements
Most training scripts require GPU support. If you only have a CPU, you can modify the script:

Python

device = 'cuda'  # Change to 'cpu'
Memory Requirements
Proteome-related tasks require significant memory. At least 32GB RAM is recommended.

File Paths
All scripts have been updated to use relative paths. Please ensure you run them from their corresponding directories:

Scripts in pressure_finetune/ must be run from that directory.

Scripts in proteome_pretrain/ must be run from that directory.

Scripts in proteome_finetune/ must be run from that directory.

FAQ
Q: Module not found error?
A: Ensure you have activated the conda environment and installed all dependencies:

Bash

conda activate scyeast
pip install -r requirements.txt
Q: Git LFS download is slow?
A: You can use a proxy or download large files individually from the release page (if available).

Q: Out of GPU memory (OOM)?
A: You can reduce the batch_size in the script or run on CPU (though it will be slower).

Citation
If you use scYeast in your research, please cite our paper:

[Citation Details to be added] (Paper citation will be added upon publication)

License
This project is open-sourced under the MIT License.

Plaintext

MIT License

Copyright (c) 2025 Hongzhong Lu (scYeast Team)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
Contact
If you have any questions, please submit a GitHub Issue or contact the project maintainers.

Last Updated: 2025-11-24
