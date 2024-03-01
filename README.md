# Brain-Scans-MQP



hehehe
print(val_fmri, "\n _ _ _ _ _ _ _ _ _ _ _ _ _ _ _\n", rh_fmri_val_pred)
linear_regression_mse = mean_squared_error(val_fmri, rh_fmri_val_pred)
print(f'Random Forest Mean Squared Error: {linear_regression_mse}')
linear_regression_mae = mean_absolute_error(val_fmri, rh_fmri_val_pred)
print(f'Random Forest Mean Absolute Error: {linear_regression_mae}')
linear_regression_r2 = r2_score(val_fmri, rh_fmri_val_pred)
print(f'Random Forest Mean R 2 Score: {linear_regression_r2}')
corr = np.corrcoef(rh_fmri_val_pred, val_fmri)
print("Corre ", np.mean(corr))
cosine = np.dot(rh_fmri_val_pred,val_fmri)/(norm(rh_fmri_val_pred)*norm(val_fmri))
print("Cosine Similarity:", cosine)
torch.cuda.empty_cache()
