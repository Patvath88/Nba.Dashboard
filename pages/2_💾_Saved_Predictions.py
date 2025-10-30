# This code should be saved as a file named 'pages/2_💾_Saved_Predictions.py'

import streamlit as st
import pandas as pd

st.title("Saved Predictions")

st.write("This page will display your saved player predictions.")

# Example of how saved predictions might be stored (using session state for simplicity)
# In a real application, you might use a database or a file to persist saved predictions
if 'saved_predictions' not in st.session_state:
    st.session_state.saved_predictions = {}

# Display saved predictions
if st.session_state.saved_predictions:
    st.subheader("Your Saved Predictions:")
    # Convert the dictionary of saved predictions to a DataFrame for display
    saved_predictions_list = []
    for player_name, prediction_data in st.session_state.saved_predictions.items():
        saved_predictions_list.append({
            'Player': player_name,
            'Predicted Points': prediction_data.get('PTS', 'N/A'),
            'Predicted Assists': prediction_data.get('AST', 'N/A'),
            'Predicted Rebounds': prediction_data.get('REB', 'N/A'),
            'Predicted 3-Pointers': prediction_data.get('FG3M', 'N/A'),
            'Season': prediction_data.get('Season', 'N/A')
            # Add other relevant prediction details here
        })
    saved_predictions_df = pd.DataFrame(saved_predictions_list)
    st.dataframe(saved_predictions_df)

    # Optional: Add functionality to clear saved predictions
    if st.button("Clear Saved Predictions"):
        st.session_state.saved_predictions = {}
        st.experimental_rerun() # Rerun to update the display
else:
    st.info("You have no saved predictions yet.")

# Note: Logic to add predictions to st.session_state.saved_predictions
# will be implemented in the main page code.
