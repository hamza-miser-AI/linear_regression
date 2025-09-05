import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data = pd.read_csv(r'C:\Users\Elite\Desktop\data mining\students.csv')
df=pd.DataFrame(data)

x=df[['age','study_hours']]
y=df['grade']
x_train, x_test, y_train, y_test =train_test_split(x , y ,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train ,y_train)

y_pred=model.predict(x_test)
print(f"coefficients:{model.coef_}")
print(f"intercept:{model.intercept_}")
print(f"mse:{mean_squared_error(y_test,y_pred)}")



# نرسم scatter
plt.scatter(df['study_hours'], df['grade'], color='blue', label='البيانات الحقيقية')

# إنشاء خط الانحدار (نثبت العمر على قيمة متوسطة مثلاً 16)
study_range = np.linspace(df['study_hours'].min(), df['study_hours'].max(), 100)
age_fixed = 16  # عمر ثابت
predicted_line = model.predict(np.column_stack([np.full_like(study_range, age_fixed), study_range]))

plt.plot(study_range, predicted_line, color='red', linewidth=2, label='خط الانحدار')

plt.xlabel('study_hours')
plt.ylabel('grade')
plt.title('العلاقة بين ساعات الدراسة ودرجة الطالب (مع ثبات العمر)')
plt.legend()
plt.show()
