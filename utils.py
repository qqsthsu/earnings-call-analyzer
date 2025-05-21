# utils.py

def generate_summary(swot_dict, entities, financials):
    summary = []

    summary.append("## Summary Report\n")

    summary.append("### Key People and Organizations")
    if entities["People Mentioned"]:
        summary.append("- People: " + ", ".join(entities["People Mentioned"]))
    if entities["Organizations Mentioned"]:
        summary.append("- Organizations: " + ", ".join(entities["Organizations Mentioned"]))

    summary.append("\n### Financial Highlights")
    if financials:
        for item in financials:
            summary.append(f"- {item}")
    else:
        summary.append("- No specific financial values were mentioned.")

    summary.append("\n### SWOT Highlights")
    for section, bullets in swot_dict.items():
        summary.append(f"\n#### {section}")
        if bullets:
            for point in bullets:
                summary.append(f"- {point}")
        else:
            summary.append("- No insights extracted.")

    return "\n".join(summary)
