import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Đặt layout của trang Streamlit thành "wide"
st.set_page_config(layout="wide")

# Thiết lập tiêu đề và subheader
# Sử dụng CSS để thay đổi màu tiêu đề
st.markdown(
    """
    <style>
    .title {
        color: #FF6347;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
        """
        <div style="text-align: center;">
            <img style="width: 90%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/18/1718719690890_9976_9475.jpeg">
        </div>
        """,
        unsafe_allow_html=True
    )
# Sử dụng Markdown để thiết lập tiêu đề với lớp CSS
# st.markdown('<h1 class="title">FINAL PROJECT</h1>', unsafe_allow_html=True)
# st.subheader("""
#              Recommender System For Coursera
#              **Quốc Khánh - Quỳnh Hiên**
#              """)
# st.write("___________________________")

menu = [
    "**Home**",
    "**Project Purpose**",
    "**Types of Recommendation Systems**",
    "**Recommender Example**",
    "**User Interaction**"
]

# CSS styles
style = """
<style>
.menu-container {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
}

.menu-item {
  margin: 0 1em;
  cursor: pointer;
}
</style>
"""

# Inject CSS styles vào ứng dụng Streamlit
st.markdown(style, unsafe_allow_html=True)

# Tạo container cho menu
menu_container = st.container()

# Sử dụng st.columns để canh giữa các radio buttons
with menu_container:
    cols = st.columns([1, 5, 1])
    
    with cols[1]:
        st.markdown('<div class="menu-container">', unsafe_allow_html=True)
        choice = st.radio("", menu, horizontal=True, key="menu_radio")
        st.markdown('</div>', unsafe_allow_html=True)


# Home

if choice == '**Home**':
    
    # Canh giữa hình ảnh bằng cách sử dụng HTML và CSS
    st.markdown(
        """
        <div style="text-align: center;">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961615570_7989_5780.jpeg">
        </div>
        """,
        unsafe_allow_html=True
    )
    
# Project Purpose

elif choice == '**Project Purpose**':
    st.markdown(
        """
        <div style="text-align: center;">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961691803_4339_5695.jpeg">
			<img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961710845_9215_3682.jpeg">
			<img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961726601_2042_7684.jpeg">
        </div>
        """,
        unsafe_allow_html=True
    )

#Types of Recommendation Systems

elif choice == '**Types of Recommendation Systems**':
    st.markdown(
        """
        <div style="text-align: center;">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961741703_7323_5838.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961757964_7454_4969.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961773814_8410_1183.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961791374_9362_9256.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961807875_6281_9446.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961838759_4100_2057.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961861859_5342_8101.jpeg">
        </div>
        """,
        unsafe_allow_html=True
    )

# Recommender Example

elif choice == '**Recommender Example**':
    st.markdown(
        """
        <div style="text-align: center;">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961878997_5264_2803.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961896160_1088_1055.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961933304_6203_6225.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961948340_7299_4219.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961967055_760_5364.jpeg">
            <img style="width: 100%" src="https://cdn-tms-supra.oviots.com/servicer/2024/06/21/1718961986348_9625_7553.jpeg">
        </div>
        """,
        unsafe_allow_html=True
    )

# User Interaction:
elif choice == "**User Interaction**": 
    logo = st.image("logo_coursera.png", width=300)
    tab1, tab2 = st.tabs(["New User", "Log In"])
    col1, col2 = tab1.columns(2) 
    
    with tab1:

        # Đọc dữ liệu từ các file CSV
        courses_df = pd.read_csv("courses_df.csv")
        
        tokenized_features = [text.split() for text in courses_df['combined_features']]

        # Train Word2Vec model
        w2v_model_new = Word2Vec(tokenized_features, vector_size=100, window=5, min_count=1, workers=4)

        # Hàm tính trung bình vector Word2Vec cho các đặc trưng kết hợp với weights
        def get_w2v_vector_new(text, weights):
            words = text.split()
            weighted_vectors = []
            for word in words:
                if word in w2v_model_new.wv:
                    weight = weights.get(word, 1)  # Lấy trọng số của từ, mặc định là 1 nếu không có trọng số
                    weighted_vectors.append(weight * w2v_model_new.wv[word])
            if not weighted_vectors:  # Nếu danh sách weighted_vectors trống, trả về vector không
                return np.zeros(w2v_model_new.vector_size)
            return np.mean(weighted_vectors, axis=0)

        # Đặt trọng số cho các từ (ví dụ)
        weights_new = {
            'course_name': 0.25,
            'unit': 0.1,
            'level': 0.1,
            'results': 0.25,
            'scaled_rating': 0.15,
            'scaled_review_number': 0.15
        }

        # Tính toán ma trận tương đồng sử dụng các vector Word2Vec với weights
        w2v_matrix_new = np.array([get_w2v_vector_new(text, weights_new) for text in courses_df['combined_features']])
        w2v_sim_new = cosine_similarity(w2v_matrix_new, w2v_matrix_new)

        # Hàm tính toán vector cho từ khóa đầu vào
        def get_keyword_vector(keyword, weights):
            return get_w2v_vector_new(keyword, weights)

        # Hàm lấy gợi ý khóa học dựa trên từ khóa
        def get_recommendations_w2v_keyword(keyword, courses_df=courses_df, w2v_sim=w2v_sim_new):
            keyword_vector = get_keyword_vector(keyword, weights_new)
            keyword_sim = cosine_similarity([keyword_vector], w2v_matrix_new)[0]
            
            # Tìm các khóa học tương tự dựa trên độ tương đồng
            sim_scores = list(enumerate(keyword_sim))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[:5]  # Lấy 5 khóa học tương tự nhất
            course_indices = [i[0] for i in sim_scores]
            return courses_df.iloc[course_indices]

        # Nhập từ khóa tìm kiếm
        keyword = st.text_input("What do you want to learn")

        # Tùy chỉnh các yếu tố gợi ý
        level = st.sidebar.selectbox("Level", courses_df["Level"].unique())
        unit = st.sidebar.selectbox("Educator", courses_df["Unit"].unique())
        rating = st.sidebar.slider("Rating", 0.0, 5.0, 4.0)
        results_limit = st.sidebar.slider("Number of recommendation", 1, 20, 10)

        # Lọc dữ liệu theo tùy chỉnh
        filtered_courses = courses_df[(courses_df["Level"] == level) &
                                    (courses_df["Unit"] == unit) &
                                    (courses_df["AvgStar"] >= rating)]

        # Hiển thị kết quả gợi ý
        if keyword:
            search_results = get_recommendations_w2v_keyword(keyword, courses_df=courses_df, w2v_sim=w2v_sim_new)
        else:
            search_results = pd.DataFrame()  # Trường hợp không có kết quả

            st.write(f"Course Recommendations:")

        # Hiển thị kết quả trên các thẻ
        cols = st.columns(3)  # Tạo các cột để chứa các thẻ kết quả

        for i, (index, row) in enumerate(search_results.iterrows()):
            # Lấy ra ngắn gọn 
            short_results = row['Results'][:50]
            col = cols[i % 3]
            with col.container():
                st.markdown(
                    f"""
                    <div style="background-color: #F5EFFB; padding: 10px; border-radius: 10px; margin: 10px 0;">
                        <h3 style="color: #29088A;">{row['CourseName']}</h3>
                        <p>Level: {row['Level']}</p>
                        <p>Educator: {row['Unit']}</p>
                        <p>Rating ⭐: {row['AvgStar']}</p>
                        <p>Skills: {short_results}</p>
                        <button>Details </button>
                </div>
                """,
                unsafe_allow_html=True
            )

    with tab2:
        # Đọc dữ liệu từ file csv
        reviews_df = pd.read_csv("reviews_sup_df.csv")
       
        # Chuẩn bị dữ liệu cho Surprise
        reader = Reader()
        data = Dataset.load_from_df(reviews_df[['Reviewer_ID', 'Course_ID', 'RatingStar']], reader)

        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        algorithm = SVD()
        algorithm.fit(trainset)
        #joblib.dump(svd, 'recommend_model.pkl') 
        # Load lại model đã trained
        #svd = joblib.load('recommend_model.pkl')

        # # Hàm gợi ý khóa học dựa trên SVD
        # def recommend_courses_svd(user_id, df, limit=20):
        #     try:
        #         user_inner_id = trainset.to_inner_uid(user_id)
        #         user_ratings = trainset.ur[user_inner_id]
        #         recommendations = []
        #         for iid, true_r in user_ratings:
        #             est = svd.predict(user_id, trainset.to_raw_iid(iid)).est
        #             recommendations.append((iid, est))
        #         recommendations.sort(key=lambda x: x[1], reverse=True)
        #         recommended_course_ids = [trainset.to_raw_iid(iid) for iid, _ in recommendations[:limit]]
        #         st.write(f"The recommended_course_ids is {recommended_course_ids}")
        #         return df.iloc[recommended_course_ids]
        #         #return df[df['CourseName'].isin(recommended_course_ids)]
        #     except ValueError:
        #         return pd.DataFrame()  # Trường hợp user_id không tồn tại trong tập huấn luyện

        # Lấy danh sách các ID từ cột Reviewer_ID
        reviewer_ids = reviews_df['Reviewer_ID'].tolist()

        # Chọn ngẫu nhiên một ID nếu chưa có trong session_state
        if "user_id" not in st.session_state:
            random_id = random.choice(reviewer_ids)
            st.session_state["user_id"] = random_id
            st.experimental_rerun()  # Tự động reload sau khi gán ID

        # Random chọn 1 ID trong danh sách
        user_id = st.text_input("Input ID")
        #user_id = st.text_input("Input ID", value=st.session_state["user_id"])
        #st.write(f"Generated ID: {user_id}")

       

        # Hiển thị kết quả gợi ý
        if user_id:
            df_select = reviews_df[(reviews_df['Reviewer_ID'] == user_id) & (reviews_df['RatingStar'] >=1)]
            df_select = df_select.set_index('Course_ID')
            df_score = reviews_df[["Course_ID", "CourseName", "RatingStar"]]
            df_score.loc[:, 'EstimateScore'] = df_score['Course_ID'].apply(lambda x: algorithm.predict(user_id, x).est)
            df_score = df_score.drop_duplicates(subset = ['Course_ID'])
            search_results = df_score.sort_values(by=['EstimateScore'], ascending=False).iloc[:results_limit]
            #search_results = reviews_df.iloc[search_results_index]
        else:
            search_results = pd.DataFrame()  # Trường hợp không có kết quả

            st.write(f"Course Recommendations:")

        # Hiển thị kết quả trên các thẻ
        cols = st.columns(3)  # Tạo các cột để chứa các thẻ kết quả

        for i, (index, row) in enumerate(search_results.iterrows()):
            col = cols[i % 3]
            with col.container():
                st.markdown(
                    f"""
                    <div style="background-color: #F5EFFB; padding: 10px; border-radius: 10px; margin: 10px 0;">
                        <h3 style="color: #29088A;">{row['CourseName']}</h3>
                        <p>Course ID: {row['Course_ID']}</p>
                        <p>Rating ⭐: {row['RatingStar']}</p>
                    </div>
                    """,
                unsafe_allow_html=True
            )
            

