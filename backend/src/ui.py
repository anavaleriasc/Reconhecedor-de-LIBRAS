"""
ui.py — Módulo dedicado à renderização avançada de interface gráfica no OpenCV.

Utiliza a biblioteca Pillow (PIL) para desenhar painéis translúcidos com
cantos arredondados, barras de progresso e textos anti-aliased usando fontes
TrueType do sistema, fornecendo uma aparência muito mais premium que as
primitivas padrão do OpenCV.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import platform

class UIRenderer:
    """
    Classe para renderização de UI complexa sobre frames do OpenCV.
    Converte o frame para PIL temporariamente para usar recursos avançados
    de desenho (alpha blending nativo, rounded rectangles, TrueType).
    """
    def __init__(self, frame_bgr: np.ndarray):
        self.height, self.width = frame_bgr.shape[:2]
        
        # Converter de BGR para RGBA para suportar desenho com alpha channel perfeitamente
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.pil_img = Image.fromarray(image_rgb).convert("RGBA")
        
        # Modo RGBA permite criar sobreposições com transparência
        self.draw = ImageDraw.Draw(self.pil_img, "RGBA")
        self.fonts = {}

    def _get_font(self, size: int, bold: bool, is_emoji: bool = False):
        key = (size, bold, is_emoji)
        if key not in self.fonts:
            system = platform.system()
            font = None
            try:
                if system == "Windows":
                    if is_emoji:
                        font_name = "seguiemj.ttf"
                    else:
                        font_name = "segoeuib.ttf" if bold else "segoeui.ttf"
                    font = ImageFont.truetype(font_name, size)
                elif system == "Darwin":
                    if is_emoji:
                        font_name = "Apple Color Emoji"
                    else:
                        font_name = "Arial Bold" if bold else "Arial"
                    font = ImageFont.truetype(font_name, size)
                else:
                    font_name = "NotoColorEmoji.ttf" if is_emoji else "DejaVuSans.ttf"
                    font = ImageFont.truetype(font_name, size)
            except IOError:
                # Fallback para fonte padrão (não escalável, pixelada)
                font = ImageFont.load_default()
            self.fonts[key] = font
        return self.fonts[key]

    def _bgr_to_rgba(self, color_bgr: tuple, alpha: int = 255) -> tuple:
        """Converte cor BGR do config para RGBA do PIL."""
        return (color_bgr[2], color_bgr[1], color_bgr[0], alpha)

    def draw_rounded_rect(self, xy: list, color_bgr: tuple, alpha: int = 150, radius: int = 10):
        """
        Desenha um retângulo com cantos arredondados e preenchimento translúcido.
        xy: [x_inicio, y_inicio, x_fim, y_fim]
        """
        color_rgba = self._bgr_to_rgba(color_bgr, alpha)
        self.draw.rounded_rectangle(xy, radius=radius, fill=color_rgba)

    def draw_rounded_rect_outline(self, xy: list, color_bgr: tuple, alpha: int = 255, radius: int = 10, width: int = 2):
        """Desenha apenas a borda de um retângulo arredondado."""
        color_rgba = self._bgr_to_rgba(color_bgr, alpha)
        self.draw.rounded_rectangle(xy, radius=radius, outline=color_rgba, width=width)

    def draw_text(self, text: str, position: tuple, color_bgr: tuple, size: int = 20, bold: bool = False, shadow: bool = True, anchor: str = "la", is_emoji: bool = False):
        """
        Desenha um texto com anti-aliasing.
        anchor: 'la' (Left-Ascender, padrão), 'mm' (Middle-Middle, centralizado), etc.
        """
        font = self._get_font(size, bold, is_emoji)
        color_rgba = self._bgr_to_rgba(color_bgr, 255)
        
        if is_emoji:
            # Emojis contêm as próprias cores nativas (COLR/CPAL)
            self.draw.text(position, text, font=font, embedded_color=True, anchor=anchor)
        else:
            if shadow:
                # Deslocamento sutil da sombra para dar profundidade
                shadow_pos = (position[0] + 2, position[1] + 2)
                self.draw.text(shadow_pos, text, font=font, fill=(0, 0, 0, 180), anchor=anchor)
                
            self.draw.text(position, text, font=font, fill=color_rgba, anchor=anchor)

    def get_text_size(self, text: str, size: int = 20, bold: bool = False) -> tuple:
        """Retorna (largura, altura) em pixels que o texto ocupará."""
        font = self._get_font(size, bold)
        # Obter bounding box do texto
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def draw_progress_bar(self, xy: list, progress: float, color_bgr: tuple = (0, 255, 0), bg_color_bgr: tuple = (50, 50, 50), alpha: int = 200):
        """
        Desenha uma barra de progresso horizontal estilo UI moderna.
        progress: float entre 0.0 e 1.0
        """
        # Limitar progresso
        progress = max(0.0, min(1.0, progress))
        
        # Fundo da barra
        self.draw_rounded_rect(xy, bg_color_bgr, alpha, radius=6)
        
        x0, y0, x1, y1 = xy
        width = x1 - x0
        inner_width = int(width * progress)
        
        if inner_width > 0:
            # Preenchimento frontal
            fg_xy = [x0, y0, x0 + inner_width, y1]
            self.draw_rounded_rect(fg_xy, color_bgr, alpha, radius=6)

    def render(self) -> np.ndarray:
        """
        Finaliza os desenhos no PIL e retorna um frame numpy em BGR,
        pronto para ser exibido no OpenCV.
        """
        # Converter de volta para RGB e depois para BGR
        result_rgb = np.array(self.pil_img.convert("RGB"))
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
