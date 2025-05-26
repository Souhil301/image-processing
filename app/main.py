import streamlit as st
from components import TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP9
from streamlit_option_menu import option_menu


st.set_page_config(page_title='Images Processing Tool', layout='wide')


with st.sidebar:
    st.title("Images Processing Tool")
    st.markdown("---")
    
    page = option_menu(
        "Navigation", 
        [
            " Home", 
            " TP1",
            " TP2",
            " TP3", 
            " TP4", 
            " TP5",
            " TP6",
            " TP7",
            " TP8",
            " TP9",


        ],
        icons=["house", "image", "sliders", "gear", "images", "graph-up", "images", "sliders", "gear", "images",], 
        menu_icon="list", 
        default_index=0
    )
    
    
if page == " Home":
    st.title("Welcome to the Images Processing Tool")
    st.write("Choose a section from the sidebar to get started.")
    st.markdown("---")
    # st.subheader("Available Features:")
    # st.write("- **TP1: ** Load and explore datasets.")
    # st.write("- **TP2: ** View statistics and visualization data")
    # st.write("- **TP3: ** Handle missing values and normalize data.")
    # st.write("- **TP4: ** Try K-Means, K-Medoids, and DBSCAN algorithms.")
    # st.write("- **TP5: ** Visualize cluster performance and metrics.")

elif page == " TP1":
    TP1.render()
elif page == " TP2":
    TP2.render()
elif page == " TP3":
    TP3.render()
    
elif page == " TP4":
    TP4.render()
    
elif page == " TP5":
    TP5.render()

elif page == " TP6":
    TP6.render()
elif page == " TP7":
    TP7.render()
elif page == " TP8":
    TP8.render()
elif page == " TP9":
    TP9.render()

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("🛠️ Built By S.MOKEDDEM")