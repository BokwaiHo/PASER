import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from scipy.sparse.linalg import eigsh
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import NMF
import networkx as nx
from rake_nltk import Rake

class PASER:
    def __init__(self, args):
        self.args = args
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rake = Rake()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_data(self, pruned_model, original_model, dataset, tokenizer):
        pruned_model.eval()
        original_model.eval()
        
        print("Performing semantic-structural clustering...")
        clusters = self.semantic_structural_clustering(dataset)
        
        print("Assessing capability degradation...")
        cluster_cds = self.assess_capability_degradation(clusters, dataset, pruned_model, original_model, tokenizer)
        
        print("Allocating budget...")
        cluster_budgets = self.allocate_budget(cluster_cds)
        
        print("Selecting samples...")
        selected_data = self.select_samples(clusters, cluster_budgets, dataset, pruned_model, original_model, tokenizer)
        
        return selected_data

    def semantic_structural_clustering(self, dataset):
        instructions = [item['instruction'] for item in dataset]
        embeddings = self.sentence_model.encode(instructions)
        diffusion_embeddings = self.apply_diffusion_kernel(embeddings)
        clusters = self.nmf_spectral_clustering(diffusion_embeddings)
        return clusters

    def apply_diffusion_kernel(self, embeddings):
        distances = pairwise_distances(embeddings)
        sigma = np.median(distances)
        A = np.exp(-distances**2 / (2 * sigma**2))
        D = np.diag(np.sum(A, axis=1))
        L = np.eye(len(embeddings)) - np.linalg.inv(D) @ A
        eigenvalues, eigenvectors = eigsh(L, k=self.args.num_clusters, which='SM')
        return eigenvectors

    def nmf_spectral_clustering(self, diffusion_embeddings):
        S = np.abs(np.dot(diffusion_embeddings, diffusion_embeddings.T))
        nmf = NMF(n_components=self.args.num_clusters, init='random', random_state=0)
        W = nmf.fit_transform(S)
        clusters = np.argmax(W, axis=1)
        return clusters

    def assess_capability_degradation(self, clusters, dataset, pruned_model, original_model, tokenizer):
        cluster_cds = {}
        for cluster_id in set(clusters):
            cluster_samples = [sample for i, sample in enumerate(dataset) if clusters[i] == cluster_id]
            cluster_cds[cluster_id] = self.compute_cds(cluster_samples, pruned_model, original_model, tokenizer)
        return cluster_cds

    def compute_cds(self, samples, pruned_model, original_model, tokenizer):
        cds = 0
        for sample in samples:
            inputs = tokenizer(sample['instruction'], return_tensors='pt').to(self.device)
            with torch.no_grad():
                pruned_output = pruned_model(**inputs)
                original_output = original_model(**inputs)
            cds += self.compute_jsd(pruned_output.logits, original_output.logits)
        return cds / len(samples)

    def compute_jsd(self, p, q):
        p = torch.softmax(p, dim=-1)
        q = torch.softmax(q, dim=-1)
        m = 0.5 * (p + q)
        return 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m))).item()

    def allocate_budget(self, cluster_cds):
        total_cds = sum(cluster_cds.values())
        return {cluster: int(self.args.max_selected_data * (cds / total_cds)) for cluster, cds in cluster_cds.items()}

    def select_samples(self, clusters, cluster_budgets, dataset, pruned_model, original_model, tokenizer):
        selected_data = []
        ccg = nx.Graph()
        
        for cluster_id, budget in cluster_budgets.items():
            cluster_samples = [sample for i, sample in enumerate(dataset) if clusters[i] == cluster_id]
            cluster_samples = sorted(cluster_samples, key=lambda x: self.compute_ies(x, pruned_model, original_model, tokenizer), reverse=True)
            
            selected_for_cluster = 0
            for sample in cluster_samples:
                if selected_for_cluster >= budget:
                    break
                
                if self.is_consistent(sample, ccg):
                    selected_data.append(sample)
                    selected_for_cluster += 1
                    self.update_ccg(sample, ccg)
        
        return selected_data

    def compute_ies(self, sample, pruned_model, original_model, tokenizer):
        inputs = tokenizer(sample['instruction'], return_tensors='pt').to(self.device)
        with torch.no_grad():
            pruned_output = pruned_model(**inputs)
            original_output = original_model(**inputs)
        jsd = self.compute_jsd(pruned_output.logits, original_output.logits)
        computational_cost = (len(inputs['input_ids'][0]) + len(sample.get('output', ''))) ** 2
        return jsd / np.log(computational_cost)

    def is_consistent(self, sample, ccg):
        concepts = self.extract_concepts(sample)
        for v1 in concepts:
            for v2 in concepts:
                if v1 != v2:
                    if v1 in ccg.nodes and v2 in ccg.nodes and not ccg.has_edge(v1, v2):
                        return False
        return True

    def update_ccg(self, sample, ccg):
        concepts = self.extract_concepts(sample)
        for v1 in concepts:
            for v2 in concepts:
                if v1 != v2:
                    ccg.add_edge(v1, v2)

    def extract_concepts(self, sample):
        text = sample['instruction'] + ' ' + sample.get('input', '') + ' ' + sample.get('output', '')
        self.rake.extract_keywords_from_text(text)
        return set(self.rake.get_ranked_phrases())
