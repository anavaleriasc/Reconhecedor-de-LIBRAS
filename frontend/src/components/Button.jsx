/**
 * Button — variantes: primary (ação principal), secondary (contorno),
 * ghost (terciária, ex.: "voltar"), danger (destrutiva, ex.: encerrar).
 */
export default function Button({
  children,
  variant = "primary",
  size,
  className = "",
  ...props
}) {
  const classes = ["btn", `btn-${variant}`, size === "sm" ? "btn-sm" : "", className]
    .filter(Boolean)
    .join(" ");

  return (
    <button className={classes} {...props}>
      {children}
    </button>
  );
}
