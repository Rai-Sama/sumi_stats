
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
        """)
        st.latex(r"b = r \frac{\sigma_y}{\sigma_x}")
        st.markdown("""
        - **a (Intercept):** predicted Y when X = 0
        """)
        st.latex(r"a = \bar{y} - b\bar{x}")
        st.markdown("""
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


'''

# app.py
"""
Business Statistics — Formula & Theory Explorer (Streamlit)
Single-file Streamlit app that presents formulae, short theory, and context
for each term. Clean UI with sidebar topic selector + expanders per formula.

Save as app.py and run: `streamlit run app.py`
"""

import streamlit as st

st.set_page_config(
    page_title="Business Statistics Formula Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Data: Topics, Theory, Formulae ----------
TOPICS = {
    "Measures of Central Tendency": {
        "theory": (
            "Central tendency measures provide a single value that is "
            "representative of the middle or centre of a distribution of data. "
            "Common measures: Mean, Median, Mode."
        ),
        "items": [
            {
                "title": "Arithmetic Mean (Ungrouped)",
                "latex": r"\bar{X} = \dfrac{\sum X}{N}",
                "terms": "ΣX = sum of all observations; N = number of observations.",
                "notes": "Use for raw (individual) data. Sensitive to outliers."
            },
            {
                "title": "Arithmetic Mean (Grouped)",
                "latex": r"\bar{X} = \dfrac{\sum fX}{\sum f}",
                "terms": "f = frequency for each class; X = class midpoint (or class mark); Σf = total frequency.",
                "notes": "Compute using mid-points when data is in class intervals."
            },
            {
                "title": "Median (Grouped)",
                "latex": r"M = L + \left(\dfrac{\frac{N}{2} - c.f.}{f}\right) \times h",
                "terms": (
                    "L = lower boundary of the median class; N = total frequency; "
                    "c.f. = cumulative frequency before median class; "
                    "f = frequency of median class; h = class width."
                ),
                "notes": "Find the class where cumulative frequency ≥ N/2 — that's the median class."
            },
            {
                "title": "Mode (Grouped)",
                "latex": r"Mode = L + \left(\dfrac{f_1 - f_0}{2f_1 - f_0 - f_2}\right) \times h",
                "terms": (
                    "L = lower limit of modal class; f₁ = frequency of modal class; "
                    "f₀ = frequency before modal class; f₂ = frequency after modal class; h = class width."
                ),
                "notes": "Use when distribution is unimodal and classes are of equal width."
            },
        ],
    },

    "Measures of Dispersion": {
        "theory": (
            "Dispersion (variability) measures show how spread out the data values are "
            "around a central value. Key measures: Range, Mean Deviation, Variance, SD, CV."
        ),
        "items": [
            {
                "title": "Range",
                "latex": r"Range = X_{\max} - X_{\min}",
                "terms": "Xmax = maximum data value; Xmin = minimum data value.",
                "notes": "Simplest measure; affected heavily by outliers."
            },
            {
                "title": "Mean Deviation (from Mean)",
                "latex": r"MD = \dfrac{\sum |X - A|}{N}",
                "terms": "A = centre (mean or median); N = number of observations.",
                "notes": "Shows average absolute deviation from central value."
            },
            {
                "title": "Standard Deviation (σ)",
                "latex": r"\sigma = \sqrt{\dfrac{\sum (X - \bar{X})^2}{N}}",
                "terms": "X = observation; 𝑋̄ = mean; N = number of observations.",
                "notes": "If using sample SD, divide by (n−1) instead of N."
            },
            {
                "title": "Coefficient of Variation (CV)",
                "latex": r"CV = \dfrac{\sigma}{\bar{X}} \times 100\%",
                "terms": "σ = standard deviation; 𝑋̄ = mean.",
                "notes": "Used to compare consistency between datasets."
            },
        ],
    },

    "Correlation Analysis": {
        "theory": (
            "Correlation measures the degree and direction of linear relationship between two variables. "
            "r ranges between −1 (perfect negative) and +1 (perfect positive)."
        ),
        "items": [
            {
                "title": "Pearson's Correlation Coefficient (r)",
                "latex": r"r = \dfrac{\sum (X - \bar{X})(Y - \bar{Y})}{\sqrt{\sum (X - \bar{X})^2 \sum (Y - \bar{Y})^2}}",
                "terms": "X, Y = variables; 𝑋̄, Ȳ = means of X, Y; Σ = summation.",
                "notes": "Measures linear association; sensitive to extreme values."
            },
            {
                "title": "Shortcut Formula for r",
                "latex": r"r = \dfrac{N\sum XY - (\sum X)(\sum Y)}{\sqrt{[N\sum X^2 - (\sum X)^2][N\sum Y^2 - (\sum Y)^2]}}",
                "terms": "N = number of pairs; ΣXY = sum of products; ΣX, ΣY, ΣX², ΣY² as usual.",
                "notes": "Easier to compute manually."
            },
            {
                "title": "Spearman’s Rank Correlation (rₛ)",
                "latex": r"r_s = 1 - \dfrac{6\sum d^2}{n(n^2 - 1)}",
                "terms": "d = rank difference for each pair; n = number of pairs.",
                "notes": "Used when data is ordinal or ranks are given."
            },
        ],
    },

    "Regression Analysis": {
        "theory": (
            "Regression is used to estimate or predict the value of one variable based on another."
        ),
        "items": [
            {
                "title": "Regression Equation of Y on X",
                "latex": r"Y = a + bX",
                "terms": "Y = dependent variable; X = independent variable; a = intercept; b = slope.",
                "notes": "Predicts Y using X."
            },
            {
                "title": "Regression Coefficient (b)",
                "latex": r"b = \dfrac{\sum (X - \bar{X})(Y - \bar{Y})}{\sum (X - \bar{X})^2}",
                "terms": "Σ = summation; 𝑋̄, Ȳ = means of X and Y.",
                "notes": "Represents rate of change of Y per unit change in X."
            },
        ],
    },

    "Probability": {
        "theory": "Probability quantifies the likelihood that an event will occur.",
        "items": [
            {
                "title": "Basic Definition",
                "latex": r"P(A) = \dfrac{\text{Favourable Outcomes}}{\text{Total Outcomes}}",
                "terms": "P(A) = probability of event A.",
                "notes": "Value lies between 0 and 1."
            },
            {
                "title": "Addition Law",
                "latex": r"P(A \cup B) = P(A) + P(B) - P(A \cap B)",
                "terms": "A ∪ B = either A or B occurs; A ∩ B = both A and B occur.",
                "notes": "Use when events are not mutually exclusive."
            },
            {
                "title": "Multiplication Law",
                "latex": r"P(A \cap B) = P(A)P(B|A)",
                "terms": "P(B|A) = conditional probability of B given A.",
                "notes": "If A, B independent → P(A∩B)=P(A)P(B)."
            },
            {
                "title": "Bayes’ Theorem",
                "latex": r"P(A_i|B) = \dfrac{P(A_i)P(B|A_i)}{\sum P(A_j)P(B|A_j)}",
                "terms": "Aᵢ = hypothesis events; B = evidence event.",
                "notes": "Used to update probabilities when new info appears."
            },
        ],
    },

    "Probability Distributions": {
        "theory": "Probability distributions describe how probabilities are distributed over possible values.",
        "items": [
            {
                "title": "Binomial Distribution",
                "latex": r"P(X=x) = {n \choose x} p^x q^{n-x}",
                "terms": "n = trials; x = successes; p = success probability; q = 1−p.",
                "notes": "Discrete distribution; used for fixed n independent trials."
            },
            {
                "title": "Normal Distribution (Z-score)",
                "latex": r"Z = \dfrac{X - \mu}{\sigma}",
                "terms": "X = data point; μ = mean; σ = standard deviation.",
                "notes": "Symmetric bell-shaped curve; used in large-sample inference."
            },
        ],
    },

    "Hypothesis Testing": {
        "theory": (
            "Hypothesis testing checks if a sample statistic significantly differs from a population parameter."
        ),
        "items": [
            {
                "title": "Z-Test (Large Samples)",
                "latex": r"Z = \dfrac{\bar{X} - \mu}{\sigma / \sqrt{n}}",
                "terms": "𝑋̄ = sample mean; μ = population mean; σ = population SD; n = sample size.",
                "notes": "Used when n>30 and σ known."
            },
            {
                "title": "t-Test (Small Samples)",
                "latex": r"t = \dfrac{\bar{X} - \mu}{s / \sqrt{n}}",
                "terms": "s = sample SD; μ = hypothesized mean; n = sample size.",
                "notes": "Used when population SD unknown."
            },
            {
                "title": "Chi-Square Test",
                "latex": r"\chi^2 = \sum \dfrac{(O - E)^2}{E}",
                "terms": "O = observed frequency; E = expected frequency.",
                "notes": "Used for categorical data goodness-of-fit or independence."
            },
        ],
    },

    "Index Numbers": {
        "theory": (
            "Index numbers measure relative change in prices, quantities, or values over time."
        ),
        "items": [
            {
                "title": "Laspeyres Index",
                "latex": r"I_L = \dfrac{\sum P_1Q_0}{\sum P_0Q_0} \times 100",
                "terms": "P₀ = base-year price; P₁ = current-year price; Q₀ = base-year quantity.",
                "notes": "Uses base-year quantities as weights."
            },
            {
                "title": "Paasche Index",
                "latex": r"I_P = \dfrac{\sum P_1Q_1}{\sum P_0Q_1} \times 100",
                "terms": "Q₁ = current-year quantity.",
                "notes": "Uses current-year quantities as weights."
            },
            {
                "title": "Fisher’s Ideal Index",
                "latex": r"I_F = \sqrt{I_L \times I_P}",
                "terms": "I_L = Laspeyres Index; I_P = Paasche Index.",
                "notes": "Considered ideal because it satisfies both time and factor reversal tests."
            },
        ],
    },

    "Time Series Analysis": {
        "theory": (
            "Time series analysis studies data over intervals of time to detect trends or seasonal effects."
        ),
        "items": [
            {
                "title": "Linear Trend (Least Squares)",
                "latex": r"Y = a + bX",
                "terms": "Y = dependent variable; X = time; a = intercept; b = trend slope.",
                "notes": "Estimates long-term direction of data."
            },
            {
                "title": "Slope of Trend Line",
                "latex": r"b = \dfrac{N\sum XY - \sum X \sum Y}{N\sum X^2 - (\sum X)^2}",
                "terms": "Σ = summation; N = number of years/periods.",
                "notes": "Used to compute regression trend line for forecasting."
            },
        ],
    },
}

# ---------- UI Layout ----------
st.title("📘 Business Statistics — Formula & Theory Explorer")
st.caption("MBA 1st Year | Welingkar Institute of Management")

with st.sidebar:
    st.header("📂 Topics")
    selected_topic = st.radio("Select a topic:", list(TOPICS.keys()))

topic_data = TOPICS[selected_topic]
st.subheader(selected_topic)
st.write(topic_data["theory"])
st.divider()

for item in topic_data["items"]:
    with st.expander(f"📘 {item['title']}", expanded=False):
        st.latex(item["latex"])
        st.markdown(f"**Terms:** {item['terms']}")
        st.markdown(f"**Notes:** {item['notes']}")

st.markdown("---")
st.caption("Created for quick revision before exams — clean, interactive formula reference.")

'''

