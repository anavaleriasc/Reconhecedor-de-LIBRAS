"""
aggregate_sessions.py — Script para compilar todos os JSONs de resumo em um único CSV.

Uso:
    python -m src.aggregate_sessions
"""

import os
import glob
import json
import csv
from backend.src import config
from backend.src.utils import garantir_diretorio

def main():
    summaries_dir = os.path.join(config.RESULTS_DIR, "analysis", "summaries")
    output_file = os.path.join(config.RESULTS_DIR, "analysis", "session_summary.csv")
    
    search_pattern = os.path.join(summaries_dir, "*.json")
    json_files = glob.glob(search_pattern)
    
    if not json_files:
        print(f"[AVISO] Nenhum arquivo JSON de resumo encontrado em {summaries_dir}")
        return
        
    print(f"[INFO] Compilando {len(json_files)} sessões...")
    
    # Colunas desejadas para o CSV final
    columns = [
        "session_name",
        "condition",
        "notes",
        "duration_seconds",
        "total_frames",
        "valid_frames",
        "hand_detection_rate",
        "most_frequent_top1",
        "average_top1_confidence",
        "average_margin",
        "class_switches",
        "temporal_stability",
        "completed_full_duration"
    ]
    
    all_data = []
    
    for json_path in sorted(json_files):
        try:
            with open(json_path, mode='r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Construir linha extraindo apenas as colunas desejadas (com fallback vazio)
            row = {col: data.get(col, "") for col in columns}
            all_data.append(row)
            
        except Exception as e:
            print(f"[ERRO] Falha ao processar {json_path}: {e}")
            
    if not all_data:
        print("[ERRO] Nenhum dado pôde ser extraído.")
        return
        
    garantir_diretorio(os.path.dirname(output_file))
    
    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"[SUCESSO] Relatório consolidado gerado em: {output_file}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar arquivo consolidado: {e}")

if __name__ == "__main__":
    main()
