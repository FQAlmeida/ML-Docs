import streamlit as st

st.title("Regression")

st.markdown("## Linear Regression")
st.markdown("### Simple Linear Regression")

st.markdown("#### Dependencies")
st.code(
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
'''
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

st.markdown("#### Dataset")

from pathlib import Path

df = pd.read_csv(Path().cwd() / "ml_docs/data/regression/Salary_Data.csv")

st.dataframe(df)

st.markdown("#### Scatter Plot")

fig, ax = plt.subplots()
ax.scatter(list(df["YearsExperience"].values), list(df["Salary"].values))
st.pyplot(fig)

st.markdown("#### Dataset Split")

from sklearn.model_selection import train_test_split
from pandas import DataFrame

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1].values, df.iloc[:, -1].values, train_size=0.7)


train_df = DataFrame({"YearsExperience": list(x_train), "Salary": list(y_train)})
test_df = DataFrame({"YearsExperience": list(x_test), "Salary": list(y_test)})

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Train Set")
    st.dataframe(train_df)
with col2:
    st.markdown("##### Test Set")
    st.dataframe(test_df)


st.markdown("#### Training")

model = LinearRegression()
model.fit(x_train, y_train)

st.markdown("#### Fit Visualizing")

fig, ax = plt.subplots()
ax.scatter(x_train, y_train, c="green", label="Train Set")
ax.scatter(x_test, y_test, c="blue", label="Test Set")
ax.plot(x_train, model.predict(x_train), c="red", label="Model")
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary vs Experience")
ax.legend(loc="best")
fig.tight_layout()
st.pyplot(fig)
