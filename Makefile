Makefile

# Target: Dependencies
#	Code

# knn.joblib knn-opt.joblib: cleaned-data.csv knn.py column_transformer.joblib
#	python knn.py

# svc.joblib svc-opt.joblib: cleaned-data.csv svc.py
#	python svc.py

# rfc.joblib rfc-opt.joblib: cleaned-data.csv rfc.py
#	python rfc.py

# nb.joblib: cleaned-data.csv nb.py
#	python nb.py

# logreg.joblib logreg-opt.joblib: cleaned-data.csv logreg.py
#	python logreg.py

# lsvc.joblib lsvc-opt.joblib: cleaned-data.csv lsvc.py
#	python lsvc.py

# cross-validation.csv: knn.joblib knn-opt.joblib svc.joblib svc-opt.joblib \
						rfc.joblib rfc-opt.joblib nb.joblib \
						logreg.joblib logreg-opt.joblib lsvc.joblib lsvc-opt.joblib
#	python cross-validation.py

# 