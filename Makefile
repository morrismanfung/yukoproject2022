# Target: Dependencies
#	Code

01-data/test.csv:
	python 02-model/dummy4make.py

02-model/01-saved-model/01-pipe_knn_opt.joblib 02-model/02-saved-scores/01-knn_dict.joblib: 02-model/03-knn.py 01-data/train.csv
	python 02-model/03-knn.py

svc-train: cleaned-data.csv 04-svc.py
	python svc.py

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