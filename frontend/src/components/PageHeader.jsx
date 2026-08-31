export default function PageHeader({
  icon: Icon,
  eyebrow,
  title,
  subtitle,
  tone = "accent",
  actions,
}) {
  return (
    <div className="page-header">
      <div className="page-header-main">
        <div className="page-header-left">
          {Icon && (
            <span className={`page-header-icon page-header-icon-${tone}`}>
              <Icon size={19} />
            </span>
          )}
          <div className="page-header-titles">
            {eyebrow && <div className="page-header-eyebrow">{eyebrow}</div>}
            <h1>{title}</h1>
            {subtitle && <p className="subtitle">{subtitle}</p>}
          </div>
        </div>
        {actions && <div className="page-header-actions">{actions}</div>}
      </div>
    </div>
  );
}
