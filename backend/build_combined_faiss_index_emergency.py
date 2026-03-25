#!/usr/bin/env python3
"""
EMERGENCY: Slim Combined FAISS Index Builder
Works around RAM thrashing by:
  1. Limiting worker pool to 4 cores (not 12+)
  2. Using 10k chunks (not 30k) 
  3. Streaming directly (no pre-loading lists)
  4. Writing metadata incrementally
  5. Aggressive garbage collection

For 16GB system that's thrashing at 85% RAM usage.

Usage:
    python3 build_combined_faiss_index_emergency.py
"""

import json
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import gc
import sys
import time
import psutil

# ============================================================================
# Setup
# ============================================================================
SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
MEDQA_JSONL = SCRIPT_DIR / "triage_formatted" / "medqa_triage.jsonl"
PMC_PARQUET = SCRIPT_DIR / "pmc_case_reports_full.parquet"

FAISS_INDEX_PATH = MODELS_DIR / "faiss_combined_medical.index"
FAISS_META_PATH = MODELS_DIR / "faiss_combined_medical_meta.jsonl"

PUBMEDBERT_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

# ============================================================================
# Main
# ============================================================================
def build():
    print("=" * 80)
    print("EMERGENCY: Slim Combined FAISS Index Builder")
    print("=" * 80)
    print("\nConfiguration:")
    print("  - Worker limit: 4 cores (vs all cores)")
    print("  - Chunk size: 10k rows (vs 30k)")
    print("  - Stream mode: Encode immediately (no pre-loading)")
    print("  - Metadata: Written incrementally")
    
    # 1. Load model ONCE (NOT spawning workers yet)
    print(f"\n[1/4] Loading PubMedBERT model...")
    try:
        model = SentenceTransformer(PUBMEDBERT_MODEL)
        model.max_seq_length = 256
        print(f"✅ Model loaded")
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
    
    # 2. Start SMALL worker pool (4 cores max)
    print(f"\n[2/4] Starting worker pool (4 cores)...")
    try:
        # CRITICAL: Limit to 4 workers to save RAM
        pool = model.start_multi_process_pool(target_devices=["cpu"] * 4)
        print(f"✅ Pool started with 4 worker processes")
    except Exception as e:
        print(f"⚠️  Pool failed: {e}. Falling back to single-thread (slower)")
        pool = None
    
    # 3. Initialize index and output files
    index = faiss.IndexFlatIP(768)
    metadata_file = open(FAISS_META_PATH, 'w')
    
    current_id = 0
    chunk_num = 0
    total_vectors = 0
    start_time = time.time()
    
    # 4. Process MedQA first (smaller, let's get a win)
    print(f"\n[3/4a] Processing MedQA JSONL...")
    
    if MEDQA_JSONL.exists():
        try:
            with open(MEDQA_JSONL, 'r') as f:
                batch_texts = []
                batch_meta = []
                batch_count = 0
                
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        case = json.loads(line)
                    except:
                        continue
                    
                    # Build text
                    text_parts = [
                        case.get('diagnosis', ''),
                        case.get('chief_complaint', ''),
                        case.get('text', '')
                    ]
                    text = ' '.join([p for p in text_parts if p]).strip()[:1000]
                    
                    batch_texts.append(text)
                    batch_meta.append({
                        'id': case.get('case_id', f'medqa_{current_id}'),
                        'source': 'medqa_triage',
                        'index': current_id
                    })
                    
                    current_id += 1
                    batch_count += 1
                    
                    # Encode in small batches
                    if batch_count >= 100:
                        try:
                            if pool:
                                embeddings = model.encode_multi_process(batch_texts, pool, batch_size=32)
                            else:
                                embeddings = model.encode(batch_texts, batch_size=32, normalize_embeddings=True)
                            
                            embeddings = embeddings.astype('float32')
                            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                            norms[norms == 0] = 1.0
                            embeddings = embeddings / norms
                            
                            index.add(embeddings)
                            total_vectors += len(embeddings)
                            
                            # Write metadata immediately
                            for m in batch_meta:
                                metadata_file.write(json.dumps(m) + '\n')
                            metadata_file.flush()
                            
                            print(f"  MedQA: Added {len(embeddings)} vectors (total: {total_vectors})")
                            
                            batch_texts.clear()
                            batch_meta.clear()
                            batch_count = 0
                            gc.collect()
                        except Exception as e:
                            print(f"  ⚠️  Batch encode failed: {e}")
                            sys.exit(1)
                
                # Flush final batch
                if batch_texts:
                    if pool:
                        embeddings = model.encode_multi_process(batch_texts, pool, batch_size=32)
                    else:
                        embeddings = model.encode(batch_texts, batch_size=32, normalize_embeddings=True)
                    
                    embeddings = embeddings.astype('float32')
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    embeddings = embeddings / norms
                    
                    index.add(embeddings)
                    total_vectors += len(embeddings)
                    
                    for m in batch_meta:
                        metadata_file.write(json.dumps(m) + '\n')
                    metadata_file.flush()
                    
                    print(f"  MedQA final: Added {len(embeddings)} vectors (total: {total_vectors})")
        
        except Exception as e:
            print(f"❌ MedQA processing failed: {e}")
            sys.exit(1)
    
    print(f"✅ MedQA complete: {total_vectors} vectors")
    
    # 5. Process PMC in 10k chunks (OPTIMIZED: no inner batching)
    print(f"\n[3/4b] Processing PMC (10k chunks)...")
    
    if not PMC_PARQUET.exists():
        print(f"❌ File not found: {PMC_PARQUET}")
        sys.exit(1)
    
    try:
        for chunk_idx, chunk in enumerate(pd.read_parquet(PMC_PARQUET, chunksize=10000)):
            chunk_num += 1
            
            # 1. Extract ALL texts from this 10k chunk at once
            texts = []
            meta_batch = []
            
            for idx, row in chunk.iterrows():
                # Use 'context' field as primary text (PMC-CaseReport dataset structure)
                text = row.get('context') or row.get('text') or row.get('content') or row.get('abstract') or ''
                if not isinstance(text, str):
                    text = str(text)
                text = text.strip()[:2000]  # Allow longer context for case reports
                
                if not text:
                    continue
                
                texts.append(text)
                
                # Use PMC_id as primary ID field (PMC-CaseReport dataset structure)
                row_id = row.get('PMC_id') or row.get('pmid') or row.get('patient_uid') or row.get('id') or f"pmc_{idx}"
                
                # Build metadata with question/answer from dataset if available
                meta_entry = {
                    'id': str(row_id),
                    'source': 'pmc_reports',
                    'index': current_id,
                    'diagnosis': row.get('diagnoses', 'N/A') if 'diagnoses' in row else 'N/A',
                    'chief_complaint': str(row.get('question', ''))[:100],  # Use question as chief complaint
                    'text_snippet': text[:200],
                    'inline': str(row.get('inline', '')) if 'inline' in row else '',
                }
                
                # Add optional fields if they exist
                if 'title' in row:
                    meta_entry['title'] = str(row.get('title', ''))[:100]
                if 'answer' in row:
                    meta_entry['answer'] = str(row.get('answer', ''))[:200]
                if 'img_ref' in row:
                    meta_entry['img_ref'] = str(row.get('img_ref', ''))
                
                meta_batch.append(meta_entry)
                
                current_id += 1
            
            if not texts:
                print(f"  ⚠️  Chunk {chunk_num}: No valid texts found, skipping")
                continue
            
            # 2. Encode the ENTIRE 10k chunk at once (much more efficient!)
            try:
                if pool:
                    # Let the pool handle internal batching with batch_size=128
                    embeddings = model.encode_multi_process(texts, pool, batch_size=128)
                else:
                    # Fallback to single-threaded with progress bar
                    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
                
                # 3. Normalize using FAISS optimized C++ implementation
                embeddings = embeddings.astype('float32')
                faiss.normalize_L2(embeddings)  # Faster than manual normalization
                
                # 4. Add all embeddings to index
                index.add(embeddings)
                total_vectors += len(embeddings)
                
                # 5. Write metadata incrementally
                for m in meta_batch:
                    metadata_file.write(json.dumps(m) + '\n')
                metadata_file.flush()
                
                # Calculate throughput and ETA
                elapsed = time.time() - start_time
                rate = total_vectors / elapsed if elapsed > 0 else 0
                eta_remaining = (326838 - total_vectors) / rate if rate > 0 else 0
                ram_percent = psutil.virtual_memory().percent
                
                print(f"  Chunk {chunk_num}: Added {len(embeddings):,} ({total_vectors:,} total | {rate:.1f} v/s | RAM: {ram_percent:.0f}% | ETA: {eta_remaining/3600:.1f}h)")
                
                # 6. Clean up memory aggressively
                del texts, meta_batch, embeddings
                gc.collect()
            
            except Exception as e:
                print(f"  ❌ Chunk {chunk_num} failed: {e}")
                sys.exit(1)
    
    except Exception as e:
        print(f"❌ PMC processing failed: {e}")
        sys.exit(1)
    
    # Close metadata file
    metadata_file.close()
    
    # 6. Cleanup and save
    print(f"\n[4/4] Finalizing...")
    
    if pool:
        try:
            model.stop_multi_process_pool(pool)
        except:
            pass
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"✅ Index saved: {FAISS_INDEX_PATH}")
    
    # Load metadata back and create lookup dictionary for frontend
    print(f"\n[5/5] Creating frontend lookup mapping...")
    lookup_dict = {}
    lookup_source_count = {'medqa_triage': 0, 'pmc_reports': 0}
    
    try:
        with open(FAISS_META_PATH, 'r') as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    meta = json.loads(line)
                    lookup_dict[str(line_idx)] = {
                        'index': line_idx,
                        'case_id': meta.get('id', f'case_{line_idx}'),
                        'source': meta.get('source', 'unknown'),
                        'diagnosis': meta.get('diagnosis', 'N/A'),
                        'chief_complaint': meta.get('chief_complaint', 'N/A'),
                        'title': meta.get('title', ''),
                        'esi_level': meta.get('esi_level', '?'),
                        'summary': meta.get('text_snippet', 'No summary available')[:200]
                    }
                    lookup_source_count[meta.get('source', 'unknown')] += 1
                except Exception as e:
                    print(f"  ⚠️  Skipping malformed line {line_idx}: {e}")
        
        # Save lookup dictionary
        lookup_path = MODELS_DIR / "faiss_combined_medical_lookup.json"
        with open(lookup_path, 'w') as f:
            json.dump(lookup_dict, f)
        
        print(f"✅ Lookup dictionary saved: {lookup_path}")
        print(f"   Total entries: {len(lookup_dict)}")
        print(f"   - MedQA triage: {lookup_source_count['medqa_triage']}")
        print(f"   - PMC reports: {lookup_source_count['pmc_reports']}")
        
    except Exception as e:
        print(f"⚠️  Lookup generation warning: {e}")
    
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    
    print("\n" + "=" * 80)
    print("✅ BUILD COMPLETE")
    print("=" * 80)
    print(f"Total vectors: {total_vectors:,}")
    print(f"Elapsed time: {hours:.2f} hours")
    print(f"Throughput: {total_vectors/elapsed:.1f} vectors/sec")
    print(f"\nOutput Files:")
    print(f"  - Index: {FAISS_INDEX_PATH}")
    print(f"  - Metadata: {FAISS_META_PATH}")
    print(f"  - Frontend Lookup: {lookup_path}")
    print(f"\nFrontend Usage:")
    print(f"  1. Load index: faiss.read_index('faiss_combined_medical.index')")
    print(f"  2. Search: distances, indices = index.search(query_vector, k=5)")
    print(f"  3. Lookup results: [lookup_dict[str(idx)] for idx in indices[0] if str(idx) in lookup_dict]")

if __name__ == "__main__":
    build()
