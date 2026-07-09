import os
import sys
import time
import json
import csv
from datetime import datetime
from collections import deque, Counter
import cv2
import numpy as np

from backend.src import config
from backend.src.utils import garantir_diretorio
from backend.src.ui import UIRenderer

class AnalysisSession:
    def __init__(self, metadata, output_dir="results/analysis"):
        self.output_dir = output_dir
        self.metadata = metadata
        self.is_recording = False
        self.start_time = None
        self.session_data = []
        self.temporal_window = deque(maxlen=15)
        self.frame_count = 0
        self.last_pred_instant = None
        self.class_switches = 0
        
        import re
        raw_name = self.metadata.get("session_name", "").strip()
        if not raw_name:
            raw_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', raw_name)
        
        # Novas propriedades exigidas
        self.session_duration_target = 10.0
        self.auto_capture_times = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.auto_captures_saved = 0
        self.completed_full_duration = False
        self.interruption_reason = ""
        
        # Preparar diretórios
        garantir_diretorio(os.path.join(self.output_dir, "logs"))
        garantir_diretorio(os.path.join(self.output_dir, "summaries"))
        self.frames_dir = os.path.join(self.output_dir, "frames", self.session_id)
        garantir_diretorio(self.frames_dir)

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording(manual=True)
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
        
        self.auto_captures_saved = 0
        self.completed_full_duration = False
        self.interruption_reason = ""
        print("\n[INFO] Gravação de sessão iniciada (Duração Alvo: 10s).")

    def stop_recording(self, manual=False):
        if not self.is_recording:
            return
        self.is_recording = False
        duration = time.time() - self.start_time
        
        if manual:
            self.completed_full_duration = False
            self.interruption_reason = "manual_stop"
            print(f"\n[INFO] Gravação INTERROMPIDA manualmente. Duração: {duration:.2f}s")
        else:
            self.completed_full_duration = True
            print(f"\n[INFO] Gravação concluída com sucesso (10s).")
            
        if len(self.session_data) > 0:
            self.save_session(duration)
        else:
            print("[AVISO] Sessão encerrada sem nenhum frame processado.")

    def get_elapsed_time(self):
        if not self.is_recording or self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def should_auto_stop(self):
        return self.is_recording and self.get_elapsed_time() >= self.session_duration_target

    def should_capture_automatically(self, elapsed_seconds):
        if self.auto_captures_saved < len(self.auto_capture_times):
            target = self.auto_capture_times[self.auto_captures_saved]
            if elapsed_seconds >= target:
                return True
        return False

    def record_frame(self, elapsed_seconds, hand_detected, handedness, top3_list, fps):
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
            "elapsed_seconds": f"{elapsed_seconds:.2f}",
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
        # Salvar CSV
        csv_path = os.path.join(self.output_dir, "logs", f"{self.session_id}.csv")
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
            "session_name": self.metadata.get("session_name", self.session_id),
            "condition": self.metadata.get("condition", "unspecified"),
            "notes": self.metadata.get("notes", ""),
            "duration_seconds": round(duration, 2),
            "session_duration_target": self.session_duration_target,
            "completed_full_duration": self.completed_full_duration,
            "total_frames": self.frame_count,
            "valid_frames": num_valid_frames,
            "hand_detection_rate": round(num_valid_frames / self.frame_count if self.frame_count > 0 else 0, 4),
            "most_frequent_top1": classe_mais_frequente,
            "top3_frequent_classes": top3_frequentes,
            "average_top1_confidence": round(confianca_media, 4),
            "average_margin": round(margem_media, 4),
            "class_switches": self.class_switches,
            "temporal_stability": round(estabilidade, 4),
            "auto_captures_target": len(self.auto_capture_times),
            "auto_capture_times_seconds": self.auto_capture_times,
            "auto_captures_saved": self.auto_captures_saved
        }
        
        if not self.completed_full_duration:
            summary["interruption_reason"] = self.interruption_reason
            
        # Salvar JSON
        json_path = os.path.join(self.output_dir, "summaries", f"{self.session_id}.json")
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

    def save_capture_frame(self, capture_type, frame_raw, frame_landmarks, elapsed_seconds, hand_detected, handedness, top3_list, smoothed):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        prefix = "auto" if capture_type == "automatic" else "manual"
        t_str = f"_t{int(elapsed_seconds):02d}" if capture_type == "automatic" else f"_{timestamp_str}"
        
        base_name = f"{prefix}_{self.session_id}{t_str}"
        img_orig_path = os.path.join(self.frames_dir, f"{base_name}_original.png")
        img_land_path = os.path.join(self.frames_dir, f"{base_name}_landmarks.png")
        json_path = os.path.join(self.frames_dir, f"{base_name}.json")
        
        try:
            cv2.imwrite(img_orig_path, frame_raw)
            cv2.imwrite(img_land_path, frame_landmarks)
        except Exception as e:
            print(f"[ERRO] Falha ao salvar imagens da captura: {e}")
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
            "capture_type": capture_type,
            "elapsed_seconds": round(elapsed_seconds, 2) if elapsed_seconds > 0 else 0.0,
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
            if capture_type == "manual":
                print(f"[INFO] Captura manual salva com sucesso: {base_name}")
            else:
                print(f"[INFO] Captura automática salva ({elapsed_seconds:.1f}s)")
        except Exception as e:
            print(f"[ERRO] Falha ao salvar JSON da captura: {e}")

def run_analysis_mode(modelo, label_encoder, camera_index: int, metadata: dict = None):
    """Executa o modo de análise avançado."""
    if metadata is None:
        metadata = {}
        
    try:
        from backend.src.landmarks import extract_hand_landmarks
        from backend.src.features import normalize_landmarks, extract_features_from_landmarks
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
    print("  F - Salvar Captura Manual atual (Original + Landmarks + JSON)")
    print("  Q - Sair")
    print("-" * 50)
    
    classes = label_encoder.classes_
    session_counter = 1
    session = AnalysisSession(metadata=metadata, output_dir=os.path.join(config.RESULTS_DIR, "analysis"))
    
    # Controle de FPS
    prev_frame_time = time.time()
    
    try:
        while True:
            ret, frame_raw_camera = cap.read()
            if not ret or frame_raw_camera is None:
                continue

            frame = cv2.flip(frame_raw_camera, 1)
            frame_raw = frame.copy() # Cópia sem desenhos para salvar no original
            
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
                        # Garantir mapeamento exato com modelo.classes_
                        encoded_classes = modelo.classes_
                        top_indices = np.argsort(probs)[::-1][:3]
                        for idx in top_indices:
                            encoded_label = encoded_classes[idx]
                            letra = label_encoder.inverse_transform([encoded_label])[0]
                            top3.append((letra, float(probs[idx])))
                    else:
                        predicao = modelo.predict(feature_vector_2d)[0]
                        letra = label_encoder.inverse_transform([predicao])[0]
                        top3.append((letra, 1.0))
                except Exception:
                    pass
            
            # Checar parada automática
            if session.should_auto_stop():
                session.stop_recording(manual=False)
                
            elapsed = session.get_elapsed_time()
            
            # Como a câmera é espelhada (cv2.flip), o MediaPipe inverte as mãos (Direita vira Esquerda).
            # Para o log, nós invertemos de volta para refletir a mão física real do usuário.
            display_handedness = None
            if handedness_label:
                display_handedness = "Right" if handedness_label == "Left" else "Left"
            
            # Registrar frame e obter predição suavizada
            smoothed_pred = session.record_frame(elapsed, hand_detected, display_handedness, top3, fps)
            
            # Checar captura automática
            if session.is_recording and session.should_capture_automatically(elapsed):
                session.save_capture_frame("automatic", frame_raw, debug_image, elapsed, hand_detected, display_handedness, top3, smoothed_pred)
                session.auto_captures_saved += 1
            
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
                if int(time.time() * 2) % 2 == 0:
                    ui.draw.ellipse((22, 22, 38, 38), fill=(255, 0, 0, 255))
                ui.draw_text(f"GRAVANDO: {elapsed:.1f}s / {session.session_duration_target}s", (45, 35), (0, 0, 255), size=14, bold=True)
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
            ui.draw_text("S: Gravar | N: Nova Sessão | F: Captura | Q: Sair", (10, altura - 25), (200, 200, 200), size=14)
                
            frame_final = ui.render()
            cv2.imshow(config.WEBCAM_WINDOW_NAME, frame_final)

            tecla = cv2.waitKey(config.WEBCAM_FPS_DELAY) & 0xFF
            
            if tecla == ord("q") or tecla == ord("Q"):
                session.stop_recording(manual=True)
                break
            elif tecla == ord("s") or tecla == ord("S"):
                session.toggle_recording()
            elif tecla == ord("n") or tecla == ord("N"):
                if session.is_recording:
                    session.stop_recording(manual=True)
                
                import tkinter as tk
                from tkinter import simpledialog
                root = tk.Tk()
                root.attributes("-topmost", True)
                root.withdraw()
                novo_nome = simpledialog.askstring("Nova Sessão", "Digite o nome da nova sessão:", parent=root)
                root.destroy()
                
                if novo_nome and novo_nome.strip():
                    new_metadata = metadata.copy()
                    new_metadata["session_name"] = novo_nome.strip()
                    metadata = new_metadata  # Atualiza para manter como base
                    session = AnalysisSession(metadata=new_metadata, output_dir=os.path.join(config.RESULTS_DIR, "analysis"))
                    print(f"\n[INFO] Nova sessão carregada e pronta: {session.session_id}")
            elif tecla == ord("f") or tecla == ord("F"):
                session.save_capture_frame("manual", frame_raw, debug_image, elapsed, hand_detected, handedness_label, top3, smoothed_pred)
                
    except KeyboardInterrupt:
        session.stop_recording(manual=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
