import os
import sys
import time
import json
import csv
from datetime import datetime
from collections import deque, Counter
import cv2
import numpy as np

from src import config
from src.utils import garantir_diretorio
from src.ui import UIRenderer

class AnalysisSession:
    def __init__(self, output_dir="results/analysis"):
        self.output_dir = output_dir
        self.is_recording = False
        self.start_time = None
        self.session_data = []
        self.temporal_window = deque(maxlen=15)
        self.frame_count = 0
        self.last_pred_instant = None
        self.class_switches = 0
        
        # Preparar diretórios
        garantir_diretorio(os.path.join(self.output_dir, "logs"))
        garantir_diretorio(os.path.join(self.output_dir, "summaries"))
        garantir_diretorio(os.path.join(self.output_dir, "frames"))

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.start_time = time.time()
        self.session_data = []
        self.frame_count = 0
        self.last_pred_instant = None
        self.class_switches = 0
        self.temporal_window.clear()
        print("\n[INFO] Gravação de sessão iniciada.")

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        duration = time.time() - self.start_time
        print(f"\n[INFO] Gravação encerrada. Duração: {duration:.2f}s")
        if len(self.session_data) > 0:
            self.save_session(duration)
        else:
            print("[AVISO] Sessão encerrada sem nenhum frame processado.")

    def record_frame(self, hand_detected, handedness, top3_list, fps):
        self.frame_count += 1
        
        # Votação temporal
        smoothed_prediction = None
        if top3_list:
            top1_class = top3_list[0][0]
            self.temporal_window.append(top1_class)
            
            if self.last_pred_instant is not None and top1_class != self.last_pred_instant:
                self.class_switches += 1
            self.last_pred_instant = top1_class
        else:
            self.temporal_window.append(None)
            
        valid_votes = [v for v in self.temporal_window if v is not None]
        if valid_votes:
            counter = Counter(valid_votes)
            smoothed_prediction = counter.most_common(1)[0][0]

        if not self.is_recording:
            return smoothed_prediction

        # Preparar dados para o log
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        top1, prob1, top2, prob2, top3, prob3 = "", "", "", "", "", ""
        margin = ""
        
        if len(top3_list) > 0:
            top1, prob1 = top3_list[0]
        if len(top3_list) > 1:
            top2, prob2 = top3_list[1]
            margin = prob1 - prob2
        if len(top3_list) > 2:
            top3, prob3 = top3_list[2]

        row = {
            "timestamp": now_str,
            "frame": self.frame_count,
            "hand_detected": hand_detected,
            "handedness": handedness if handedness else "",
            "top1": top1,
            "prob_top1": f"{prob1:.4f}" if isinstance(prob1, float) else "",
            "top2": top2,
            "prob_top2": f"{prob2:.4f}" if isinstance(prob2, float) else "",
            "top3": top3,
            "prob_top3": f"{prob3:.4f}" if isinstance(prob3, float) else "",
            "margin_top1_top2": f"{margin:.4f}" if isinstance(margin, float) else "",
            "smoothed_prediction": smoothed_prediction if smoothed_prediction else "",
            "fps": f"{fps:.1f}"
        }
        self.session_data.append(row)
        
        return smoothed_prediction

    def save_session(self, duration):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salvar CSV
        csv_path = os.path.join(self.output_dir, "logs", f"analysis_{timestamp_str}.csv")
        try:
            with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                if not self.session_data:
                    return
                fieldnames = self.session_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.session_data)
            print(f"[INFO] Log CSV salvo em: {csv_path}")
        except Exception as e:
            print(f"[ERRO] Falha ao salvar CSV: {e}")
            
        # Calcular Resumo Matemático
        valid_frames = [r for r in self.session_data if r["hand_detected"]]
        num_valid_frames = len(valid_frames)
        
        top1_classes = [r["top1"] for r in valid_frames if r["top1"]]
        top1_counter = Counter(top1_classes)
        top3_frequentes = top1_counter.most_common(3)
        
        classe_mais_frequente = top3_frequentes[0][0] if top3_frequentes else None
        frequencia_maxima = top3_frequentes[0][1] if top3_frequentes else 0
        estabilidade = (frequencia_maxima / num_valid_frames) if num_valid_frames > 0 else 0
        
        probs_top1 = [float(r["prob_top1"]) for r in valid_frames if r["prob_top1"]]
        confianca_media = sum(probs_top1) / len(probs_top1) if probs_top1 else 0
        
        margins = [float(r["margin_top1_top2"]) for r in valid_frames if r["margin_top1_top2"] != ""]
        margem_media = sum(margins) / len(margins) if margins else 0
        
        summary = {
            "duration_seconds": round(duration, 2),
            "total_frames": self.frame_count,
            "valid_frames": num_valid_frames,
            "hand_detection_rate": round(num_valid_frames / self.frame_count if self.frame_count > 0 else 0, 4),
            "most_frequent_top1": classe_mais_frequente,
            "top3_frequent_classes": top3_frequentes,
            "average_top1_confidence": round(confianca_media, 4),
            "average_margin": round(margem_media, 4),
            "class_switches": self.class_switches,
            "temporal_stability": round(estabilidade, 4)
        }
        
        # Salvar JSON
        json_path = os.path.join(self.output_dir, "summaries", f"analysis_{timestamp_str}.json")
        try:
            with open(json_path, mode='w', encoding='utf-8') as f:
                json.dump(summary, f, indent=4)
            print(f"[INFO] Resumo JSON salvo em: {json_path}")
        except Exception as e:
            print(f"[ERRO] Falha ao salvar JSON: {e}")
            
        print("\n--- RESUMO DA SESSÃO ---")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("------------------------\n")

    def save_problematic_frame(self, frame_bgr, hand_detected, handedness, top3_list, smoothed):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        img_path = os.path.join(self.output_dir, "frames", f"frame_{timestamp_str}.png")
        json_path = os.path.join(self.output_dir, "frames", f"frame_{timestamp_str}.json")
        
        try:
            cv2.imwrite(img_path, frame_bgr)
        except Exception as e:
            print(f"[ERRO] Falha ao salvar imagem do frame problemático: {e}")
            return
            
        top1, prob1, top2, prob2, top3, prob3 = "", "", "", "", "", ""
        margin = ""
        
        if len(top3_list) > 0:
            top1, prob1 = top3_list[0]
        if len(top3_list) > 1:
            top2, prob2 = top3_list[1]
            margin = prob1 - prob2
        if len(top3_list) > 2:
            top3, prob3 = top3_list[2]
            
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hand_detected": hand_detected,
            "handedness": handedness,
            "top1": top1,
            "prob_top1": round(prob1, 4) if isinstance(prob1, float) else "",
            "top2": top2,
            "prob_top2": round(prob2, 4) if isinstance(prob2, float) else "",
            "top3": top3,
            "prob_top3": round(prob3, 4) if isinstance(prob3, float) else "",
            "margin_top1_top2": round(margin, 4) if isinstance(margin, float) else "",
            "smoothed_prediction": smoothed
        }
        
        try:
            with open(json_path, mode='w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"[INFO] Frame problemático salvo com sucesso: {timestamp_str}")
        except Exception as e:
            print(f"[ERRO] Falha ao salvar JSON do frame problemático: {e}")

def run_analysis_mode(modelo, label_encoder, camera_index: int):
    """Executa o modo de análise avançado."""
    try:
        from src.landmarks import extract_hand_landmarks
        from src.features import normalize_landmarks, extract_features_from_landmarks
    except ImportError as e:
        print(f"Erro ao carregar os módulos core: {e}")
        sys.exit(1)

    print(f"\nAbrindo webcam no modo Análise Avançado (índice {camera_index})...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Erro: não foi possível acessar a webcam.")
        sys.exit(1)

    print("\nControles do Modo de Análise:")
    print("  S - Iniciar/Parar Gravação da Sessão (CSV + JSON)")
    print("  F - Salvar Frame Problemático atual (Imagem + JSON)")
    print("  Q - Sair")
    print("-" * 50)
    
    classes = label_encoder.classes_
    session = AnalysisSession(output_dir=os.path.join(config.RESULTS_DIR, "analysis"))
    
    # Controle de FPS
    prev_frame_time = time.time()
    
    try:
        while True:
            ret, frame_raw = cap.read()
            if not ret or frame_raw is None:
                continue

            frame = cv2.flip(frame_raw, 1)
            
            # Calcular FPS
            new_frame_time = time.time()
            dt = new_frame_time - prev_frame_time
            fps = 1 / dt if dt > 0 else 0
            prev_frame_time = new_frame_time
            
            top3 = []
            hand_detected = False
            handedness_label = None
            
            try:
                landmarks_array, debug_image, success, handedness_label = extract_hand_landmarks(frame)
            except Exception:
                landmarks_array, debug_image, success, handedness_label = None, frame, False, None

            if success and landmarks_array is not None:
                hand_detected = True
                try:
                    normalized = normalize_landmarks(landmarks_array, handedness_label)
                    feature_vector = extract_features_from_landmarks(normalized)
                    
                    feature_vector_2d = feature_vector.reshape(1, -1)
                    if hasattr(modelo, "predict_proba"):
                        probs = modelo.predict_proba(feature_vector_2d)[0]
                        top_indices = np.argsort(probs)[::-1][:3]
                        for idx in top_indices:
                            top3.append((classes[idx], probs[idx]))
                    else:
                        predicao = modelo.predict(feature_vector_2d)[0]
                        letra = label_encoder.inverse_transform([predicao])[0]
                        top3.append((letra, 1.0))
                except Exception:
                    pass
            
            # Registrar frame e obter predição suavizada
            smoothed_pred = session.record_frame(hand_detected, handedness_label, top3, fps)
            
            # =========================================================
            # Renderizar HUD Avançado
            # =========================================================
            ui = UIRenderer(debug_image)
            altura, largura = debug_image.shape[:2]
            
            # Painel esquerdo
            hud_bg = [10, 10, 320, 250]
            ui.draw_rounded_rect(hud_bg, (20, 20, 20), alpha=230, radius=12)
            ui.draw_rounded_rect_outline(hud_bg, (100, 100, 100), alpha=255, radius=12, width=1)
            
            # Indicador de gravação
            if session.is_recording:
                # Piscar bolinha vermelha
                if int(time.time() * 2) % 2 == 0:
                    ui.draw.ellipse((22, 22, 38, 38), fill=(255, 0, 0, 255))
                ui.draw_text("GRAVANDO SESSÃO...", (45, 35), (0, 0, 255), size=14, bold=True)
            else:
                ui.draw_text("MODO ANÁLISE", (30, 35), (255, 255, 255), size=14, bold=True)
                
            ui.draw_text(f"FPS: {fps:.0f}", (240, 35), (150, 150, 150), size=12)
            
            # Seção de Predição
            ui.draw.line((20, 60, 310, 60), fill=(100, 100, 100, 255), width=1)
            
            y_offset = 75
            if hand_detected and len(top3) > 0:
                top1_letra, prob1 = top3[0]
                
                # Predição Instantânea vs Suavizada
                ui.draw_text(f"Instantânea: {top1_letra}", (20, y_offset), (200, 200, 200), size=16)
                ui.draw_text(f"Suavizada: {smoothed_pred if smoothed_pred else '-'}", (180, y_offset), (0, 255, 255), size=16, bold=True)
                
                ui.draw.line((20, y_offset + 30, 310, y_offset + 30), fill=(100, 100, 100, 255), width=1)
                y_offset += 40
                
                # Top 3
                for i, (letra, prob) in enumerate(top3):
                    cor = (0, 255, 0) if i == 0 else (150, 150, 150)
                    ui.draw_text(f"{i+1}. {letra}", (20, y_offset), cor, size=20, bold=(i==0))
                    ui.draw_text(f"{prob*100:.1f}%", (80, y_offset), cor, size=20, bold=(i==0))
                    y_offset += 30
                    
                # Margem de incerteza
                if len(top3) > 1:
                    margem = prob1 - top3[1][1]
                    cor_margem = (0, 0, 255) if margem < 0.1 else (0, 200, 255) if margem < 0.3 else (0, 255, 0)
                    ui.draw_text(f"Margem (1º - 2º): {margem*100:.1f}%", (20, y_offset + 5), cor_margem, size=14)
            else:
                ui.draw_text("Mão não detectada", (20, y_offset), (0, 0, 255), size=16)

            # Legenda de teclas no rodapé
            ui.draw_text("S: Gravar | F: Salvar Frame | Q: Sair", (10, altura - 25), (200, 200, 200), size=14)
                
            frame_final = ui.render()
            cv2.imshow(config.WEBCAM_WINDOW_NAME, frame_final)

            tecla = cv2.waitKey(config.WEBCAM_FPS_DELAY) & 0xFF
            
            if tecla == ord("q") or tecla == ord("Q"):
                session.stop_recording()
                break
            elif tecla == ord("s") or tecla == ord("S"):
                session.toggle_recording()
            elif tecla == ord("f") or tecla == ord("F"):
                session.save_problematic_frame(debug_image, hand_detected, handedness_label, top3, smoothed_pred)
                
    except KeyboardInterrupt:
        session.stop_recording()
    finally:
        cap.release()
        cv2.destroyAllWindows()
