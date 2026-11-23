import json, os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Inference Lab", layout="wide")
st.title("Inference Lab Results")

# Sidebar settings
GPU_PRICE = st.sidebar.number_input("GPU price per hour (USD)", value=0.50, step=0.05)
CURRENCIES = {"USD": 1.0, "EUR": 0.86, "GBP": 0.76, "CHF": 0.80, "AED": 3.67}  # approximate rates
CODE = st.sidebar.selectbox("Currency", list(CURRENCIES))
RATE = st.sidebar.number_input(f"Exchange rate: 1 USD = ? {CODE}", value=CURRENCIES[CODE], key=f"rate_{CODE}")
N_REQUESTS = 40  # requests used in the batching test

def load(name):
    return json.load(open(name, encoding="utf-8")) if os.path.exists(name) else None

quant = load("quant_results.json")
batch = load("batching_results.json")
cache = load("caching_results.json")
stream = load("streaming_results.json")
speculative = load("speculative_results.json")
percentiles = load("latency_percentiles.json")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Quantization", 
    "Batching", 
    "Caching", 
    "Streaming", 
    "Speculative Decoding", 
    "Tail Latency (P95/P99)", 
    "Report"
])

# Tab 1: Quantization
with tab1:
    if quant:
        df = pd.DataFrame(quant)
        COST1 = f"cost_per_1M_tokens_{CODE}"
        df[COST1] = (GPU_PRICE * RATE / (df["tokens_per_s"] * 3600) * 1_000_000).round(2)
        st.dataframe(df, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.subheader("Tokens per second"); c1.bar_chart(df.set_index("version")["tokens_per_s"])
        c2.subheader("Latency (s)"); c2.bar_chart(df.set_index("version")["latency_s"])
        c3.subheader(f"Cost per 1M tokens ({CODE})"); c3.bar_chart(df.set_index("version")[COST1])
    else:
        st.warning("quant_results.json not found")

# Tab 2: Batching
with tab2:
    if batch:
        df = pd.DataFrame(batch)
        COST2 = f"cost_per_request_{CODE}"
        df[COST2] = (GPU_PRICE * RATE * df["total_time_s"] / 3600 / N_REQUESTS).round(7)
        st.dataframe(df, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.subheader("Tokens per second"); c1.line_chart(df.set_index("concurrency")["tokens_per_s"])
        c2.subheader("Avg latency (s)"); c2.line_chart(df.set_index("concurrency")["avg_latency_s"])
        c3.subheader(f"Cost per request ({CODE})"); c3.line_chart(df.set_index("concurrency")[COST2])
    else:
        st.warning("batching_results.json not found")

# Tab 3: Prefix Caching
with tab3:
    if cache:
        df = pd.DataFrame(cache)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("prefix_caching")["avg_ttft_after_first_s"])
    else:
        st.warning("caching_results.json not found")

# Tab 4: Streaming
with tab4:
    if stream:
        df = pd.DataFrame(stream).T
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["first_visible_s"])
    else:
        st.warning("streaming_results.json not found")

# Tab 5: Speculative Decoding (New)
with tab5:
    if speculative:
        df = pd.DataFrame(speculative)
        st.subheader("Baseline vs Speculative Decoding Comparison")
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.subheader("Throughput (Tokens / s)")
        c1.bar_chart(df.set_index("name")["tokens_per_s"])
        c2.subheader("Latency (Seconds)")
        c2.bar_chart(df.set_index("name")["latency_s"])
    else:
        st.warning("speculative_results.json not found")

# Tab 6: Tail Latency Percentiles (New)
with tab6:
    if percentiles:
        st.subheader(f"Latency Percentiles (Concurrency: {percentiles.get('concurrency', 'N/A')}, Requests: {percentiles.get('total_requests', 'N/A')})")
        
        # Build comparison table
        ttft_data = percentiles.get("ttft_seconds", {})
        lat_data = percentiles.get("total_latency_seconds", {})
        
        df_p = pd.DataFrame({
            "Metric": ["Mean (Average)", "p50 (Median)", "p90", "p95", "p99"],
            "TTFT (Seconds)": [ttft_data.get("mean"), ttft_data.get("p50"), ttft_data.get("p90"), ttft_data.get("p95"), ttft_data.get("p99")],
            "Total Latency (Seconds)": [lat_data.get("mean"), lat_data.get("p50"), lat_data.get("p90"), lat_data.get("p95"), lat_data.get("p99")]
        }).set_index("Metric")
        
        st.dataframe(df_p, use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.subheader("Time-to-First-Token (TTFT) Tail (s)")
        c1.bar_chart(df_p["TTFT (Seconds)"])
        c2.subheader("Total Latency Tail (s)")
        c2.bar_chart(df_p["Total Latency (Seconds)"])
    else:
        st.warning("latency_percentiles.json not found")

# Tab 7: Comprehensive Report
with tab7:
    lines = ["# Inference Lab Report", "", "Model: Qwen2.5-1.5B-Instruct, served with vLLM on GPU.", ""]
    if quant:
        best = max(quant, key=lambda r: r["tokens_per_s"])
        lines.append(f"- **Quantization:** {best['version']} was fastest ({best['tokens_per_s']} tokens/s, accuracy {best['accuracy']:.0%}).")
    if batch:
        lo, hi = batch[0], batch[-1]
        lines.append(f"- **Batching:** throughput rose from {lo['tokens_per_s']} to {hi['tokens_per_s']} tokens/s "
                     f"at concurrency {hi['concurrency']}, while average latency went from {lo['avg_latency_s']}s to {hi['avg_latency_s']}s.")
    if cache:
        off = next(r for r in cache if not r["prefix_caching"])
        on = next(r for r in cache if r["prefix_caching"])
        lines.append(f"- **Prefix caching:** time to first token after the first request dropped from "
                     f"{off['avg_ttft_after_first_s']}s to {on['avg_ttft_after_first_s']}s.")
    if stream:
        lines.append(f"- **Streaming:** first text appeared after {stream['streaming']['first_visible_s']}s instead of "
                     f"{stream['non_streaming']['first_visible_s']}s, with the same total time.")
    if speculative:
        spec_text = ", ".join(f"{item['name']}: {item['tokens_per_s']} tps" for item in speculative)
        lines.append(f"- **Speculative Decoding:** evaluated ({spec_text}).")
    if percentiles:
        p95 = percentiles.get("total_latency_seconds", {}).get("p95")
        p50 = percentiles.get("total_latency_seconds", {}).get("p50")
        lines.append(f"- **Tail Latency:** median response time was {p50}s, but p95 spiked to {p95}s under concurrency.")
    lines.append(f"- **Cost assumption:** GPU price of ${GPU_PRICE}/hour, shown in {CODE} at 1 USD = {RATE} {CODE}.")
    report = "\n".join(lines)
    st.markdown(report)
    st.download_button("Download report.md", report, file_name="report.md")