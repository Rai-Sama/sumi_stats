import streamlit as st
import pandas as pd

st.set_page_config(page_title="Business Statistics Formula Sheet", layout="centered")

st.title("📘 Business Statistics – Numerical Formula & Example Sheet (MBA E-Business)")
st.write("Each section includes formulas, use-cases, and explanations for solving numericals effectively.")

# -----------------------------------------------------------
# 1️⃣ Measures of Central Tendency & Dispersion
# -----------------------------------------------------------
with st.expander("1️⃣ Measures of Central Tendency & Dispersion"):
    st.markdown("""
Used to **summarize data numerically** — averages show where most data lies, and dispersion shows how spread out data is.
    """)

    st.subheader("👉 Central Tendency (Average or Representative Value)")
    st.latex(r"\bar{x} = \frac{\sum x}{n}")
    st.caption("Arithmetic Mean – the most common average; sensitive to extreme values.")

    st.latex(r"\bar{x}_w = \frac{\sum wx}{\sum w}")
    st.caption("Weighted Mean – gives weights to data points (e.g., average marks with subject weightage).")

    st.latex(r"Median = L + \frac{(N/2 - CF)}{f} \times h")
    st.caption("Median – divides ordered data into two equal halves; useful when data has outliers.")

    st.latex(r"Mode = L + \frac{(f_m - f_1)}{(2f_m - f_1 - f_2)} \times h")
    st.caption("Mode – most frequent value or class (used in categorical or grouped data).")

    st.subheader("👉 Dispersion (Spread of Data)")
    st.latex(r"Variance = \frac{\sum (x - \bar{x})^2}{n}")
    st.caption("Variance – measures average squared deviation from the mean.")

    st.latex(r"SD = \sqrt{Variance}")
    st.caption("Standard Deviation – average deviation from the mean (square root of variance).")

    st.latex(r"CV = \frac{SD}{Mean} \times 100")
    st.caption("Coefficient of Variation – useful for comparing variability between datasets of different scales.")

# -----------------------------------------------------------
# 2️⃣ Probability
# -----------------------------------------------------------
with st.expander("2️⃣ Probability"):
    st.markdown("""
Used to measure **how likely an event is to occur** — the foundation of inferential statistics.
    """)

    st.latex(r"P(A \cup B) = P(A) + P(B) - P(A \cap B)")
    st.caption("Union Rule – probability of A or B happening (avoids double-counting overlap).")

    st.latex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}")
    st.caption("Conditional Probability – chance of A occurring when B has already occurred.")

    st.latex(r"P(A_i|B) = \frac{P(B|A_i)P(A_i)}{\sum P(B|A_j)P(A_j)}")
    st.caption("Bayes’ Theorem – updates prior probability after observing evidence.")

# -----------------------------------------------------------
# 3️⃣ Distributions
# -----------------------------------------------------------
with st.expander("3️⃣ Distributions"):
    st.markdown("""
Distributions describe how probabilities or frequencies are **distributed across all possible outcomes**.
    """)

    st.subheader("Discrete Distributions")
    st.latex(r"P(X=k) = {n \choose k} p^k (1-p)^{n-k}")
    st.caption("Binomial – models fixed number of independent yes/no trials (e.g., 5 coin tosses).")

    with st.expander("More context on Binomial distribution"):
        st.markdown("""
        - **n:** number of trials  
        - **p:** probability of success  
        - **k:** number of successes  
        - Example: Probability of getting 3 heads in 5 tosses = Binomial(5, 0.5)
        """)

    st.latex(r"P(X=k) = e^{-\lambda} \frac{\lambda^k}{k!}")
    st.caption("Poisson – models rare events in fixed intervals (e.g., incoming calls per minute).")

    with st.expander("More context on Poisson distribution"):
        st.markdown("""
        - **λ (lambda):** average rate of occurrence  
        - Used when n is large and p is small (n·p = λ).  
        - Example: If λ=3 (avg 3 arrivals/min), P(X=2) = 0.224.
        """)

    st.subheader("Continuous Distribution")
    st.latex(r"z = \frac{x - \mu}{\sigma}")
    st.caption("Normal – converts values to standard Z-scores to find probabilities using tables.")

    with st.expander("More context on Normal distribution"):
        st.markdown("""
        - **μ:** population mean  
        - **σ:** population standard deviation  
        - Z-score shows how many SDs a value is from the mean.  
        - Used for probability of ranges, e.g., P(70 < X < 90).
        """)

# -----------------------------------------------------------
# 4️⃣ Sampling & Estimation
# -----------------------------------------------------------
with st.expander("4️⃣ Sampling & Estimation"):
    st.markdown("""
Sampling allows conclusions about a population from a smaller group. Estimation predicts unknown parameters.
    """)

    st.latex(r"SE = \frac{\sigma}{\sqrt{n}}")
    st.caption("Standard Error – measures variation in sample means; lower SE = more reliable estimate.")

    st.latex(r"CI = \bar{x} \pm z \times \frac{\sigma}{\sqrt{n}}")
    st.caption("Confidence Interval – gives a range of plausible values for population mean.")

    with st.expander("More context on Confidence Intervals"):
        st.markdown("""
        - **z:** critical value (1.96 for 95%, 2.58 for 99%)  
        - **σ:** population SD (if unknown, use sample SD and t-distribution).  
        - Example: CI(95%) for sample mean 50, σ=10, n=25 → (46.08, 53.92)
        """)

# -----------------------------------------------------------
# 5️⃣ Hypothesis Testing
# -----------------------------------------------------------
with st.expander("5️⃣ Hypothesis Testing"):
    st.markdown("""
Used to **check claims about population parameters** using sample evidence.
    """)

    st.latex(r"z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}")
    st.caption("Z-test – used when population SD is known or n > 30.")

    st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")
    st.caption("t-test – used for small samples (n < 30) when σ is unknown.")

    st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
    st.caption("Chi-Square – tests independence between categorical variables or goodness of fit.")

    with st.expander("More context on Chi-Square"):
        st.markdown("""
        - **O:** observed frequency  
        - **E:** expected frequency  
        - Larger χ² → greater difference between observed and expected → possible rejection of null hypothesis.
        """)

    # ------- ANOVA main content + deeper nested expanders
    st.latex(r"F = \frac{MSB}{MSW}")
    st.caption("ANOVA compares means of 3 or more groups via ratio of between-group to within-group variance.")

    with st.expander("ANOVA — definitions & formulas (SSB, SSW, SST, df, MS)"):
        st.write("**Symbols & what they mean:**")
        st.latex(r"\text{Let } k \text{ = \# of groups, } n_i \text{ = sample size of group } i, \; N=\sum_i n_i")
        st.latex(r"\bar{x}_i \text{ = mean of group } i,\quad \bar{x} \text{ = overall (grand) mean}")
        st.markdown("**Sum of Squares (definitions):**")
        st.latex(r"SSB = \sum_{i=1}^{k} n_i (\bar{x}_i - \bar{x})^2 \quad\text{(Sum of Squares Between groups)}")
        st.caption("SSB measures how far each group mean is from the overall mean — captures between-group variation.")
        st.latex(r"SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2 \quad\text{(Sum of Squares Within groups)}")
        st.caption("SSW measures variability of observations within each group — captures within-group variation (error).")
        st.latex(r"SST = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (x_{ij} - \bar{x})^2 = SSB + SSW")
        st.caption("Total variation in the data; SST partitions into between + within components.")
        st.markdown("**Degrees of freedom & Mean Squares:**")
        st.latex(r"df_{between} = k - 1\quad,\quad df_{within} = N - k")
        st.latex(r"MSB = \frac{SSB}{k - 1}\quad,\quad MSW = \frac{SSW}{N - k}")
        st.caption("F statistic = MSB / MSW. Compare to F_{critical}(df_between, df_within) or compute p-value.")

        with st.expander("Worked numeric example — step-by-step (computed live)"):
            st.write("We compute an ANOVA table for 3 groups (each n=3):")
            st.write("Group A = [10, 12, 9]; Group B = [15, 16, 14]; Group C = [9, 8, 10].")

            # data & computations
            groups = {
                "A": [10, 12, 9],
                "B": [15, 16, 14],
                "C": [9, 8, 10]
            }
            # compute group sizes, means
            group_names = list(groups.keys())
            n_i = {k: len(v) for k, v in groups.items()}
            mean_i = {k: sum(v) / len(v) for k, v in groups.items()}
            N = sum(n_i.values())
            grand_mean = sum(sum(v) for v in groups.values()) / N

            # compute SSB and SSW with per-group details
            ssb_components = {}
            for k, v in groups.items():
                ssb_components[k] = n_i[k] * (mean_i[k] - grand_mean) ** 2
            SSB = sum(ssb_components.values())

            ssw_components = {}
            for k, v in groups.items():
                ssw_components[k] = sum((x - mean_i[k]) ** 2 for x in v)
            SSW = sum(ssw_components.values())

            SST = SSB + SSW
            k = len(groups)
            df_between = k - 1
            df_within = N - k
            MSB = SSB / df_between
            MSW = SSW / df_within
            F_stat = MSB / MSW

            # display computed numbers
            st.write("**Computed group means & sizes**")
            gm_df = pd.DataFrame({
                "Group": group_names,
                "n_i": [n_i[g] for g in group_names],
                "mean_i": [mean_i[g] for g in group_names],
                "SSB_component": [ssb_components[g] for g in group_names],
                "SSW_component": [ssw_components[g] for g in group_names],
            })
            gm_df["mean_i"] = gm_df["mean_i"].map(lambda x: round(x, 6))
            gm_df["SSB_component"] = gm_df["SSB_component"].map(lambda x: round(x, 6))
            gm_df["SSW_component"] = gm_df["SSW_component"].map(lambda x: round(x, 6))
            st.table(gm_df)

            st.write("**Overall (grand) mean & totals**")
            st.markdown(f"- Grand mean (\\bar{{x}}) = {grand_mean:.6f}")
            st.markdown(f"- SSB = sum of SSB components = {SSB:.6f}")
            st.markdown(f"- SSW = sum of SSW components = {SSW:.6f}")
            st.markdown(f"- SST = SSB + SSW = {SST:.6f}")

            # ANOVA table as DataFrame
            anova_df = pd.DataFrame([
                ["Between Groups", round(SSB, 6), df_between, round(MSB, 6), round(F_stat, 6)],
                ["Within Groups", round(SSW, 6), df_within, round(MSW, 6), ""],
                ["Total", round(SST, 6), df_between + df_within, "", ""],
            ], columns=["Source", "SS", "df", "MS", "F"])
            st.subheader("ANOVA table")
            st.table(anova_df)

            st.write("**Interpretation:**")
            st.write(
                f"- F = {F_stat:.4f} with df_between = {df_between}, df_within = {df_within}.\n"
                "- If F > F_critical (from F-table) or p-value < α, reject H0 (means are not all equal)."
            )


# -----------------------------------------------------------
# 6️⃣ Correlation & Regression
# -----------------------------------------------------------
with st.expander("6️⃣ Correlation & Regression"):
    st.markdown("""
Used to **analyze and predict relationships** between quantitative variables.
    """)

    st.latex(r"r = \frac{n\Sigma xy - \Sigma x \Sigma y}{\sqrt{(n\Sigma x^2 - (\Sigma x)^2)(n\Sigma y^2 - (\Sigma y)^2)}}")
    st.caption("Correlation (r) – shows how strongly X and Y move together (–1 to +1).")

    with st.expander("More context on Correlation"):
        st.markdown("""
        - **r > 0:** positive relationship (X↑ → Y↑)  
        - **r < 0:** negative relationship (X↑ → Y↓)  
        - **r = 0:** no linear relation  
        - Sensitive to outliers.
        """)

    st.latex(r"Y = a + bX")
    st.caption("Simple Linear Regression – predicts Y using X.")

    with st.expander("More context on Regression"):
        st.markdown("""
        - **b (Slope):** rate of change in Y for one-unit change in X  
          \\( b = r \frac{\sigma_y}{\sigma_x} \\)
        - **a (Intercept):** predicted Y when X = 0  
          \\( a = \bar{y} - b\bar{x} \\)
        - Regression helps in prediction and quantifying impact of X on Y.
        """)

    st.latex(r"R^2 = r^2")
    st.caption("Coefficient of Determination – proportion of variance in Y explained by X.")

# -----------------------------------------------------------
# Tip Section
# -----------------------------------------------------------
st.info("""
💡 **Quick Tools for Practice:**
- Use Excel formulas: `AVERAGE`, `STDEV`, `BINOM.DIST`, `NORM.DIST`, `CORREL`, `LINEST`
- Use **Data Analysis ToolPak** for ANOVA, Regression, and t-tests.
- For practice, create small 5–10 record datasets and test each formula manually and via Excel.
""")
