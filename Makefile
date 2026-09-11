# ----------------
#  Arguments
# ----------------
DATA_DIR := $(or $(DATA_DIR), ../data)

# ----------------
# Definitions
# ----------------
datasets_list := $(or $(DS_LIST), CIFAR100 ImageNet)
models := $(or $(MODELS), VGG MobileNet ResNet ConvNeXt)
reductions := $(or $(REDUCTIONS), kernel toeplitz avgpooling)
analyses := $(or $(ANALYSES), MACS DMD)

# ----------------
# Parsing names
# ----------------
datasets := $(foreach ds, $(datasets_list), $(foreach m, $(models), $(DATA_DIR)/$(ds)/datasets/dss.$(ds)-test.APGD-t-$(m)))

corevectors := $(foreach ds, $(datasets_list), $(foreach p, $(foreach m, $(models), $(DATA_DIR)/$(ds)/$(m)/corevectors), $(foreach r, $(reductions), $(p)/$(r))))

tunings := $(foreach ds, $(datasets_list), $(foreach pp, $(foreach p, $(foreach m, $(models), $(DATA_DIR)/$(ds)/$(m)/peepholes), $(foreach r, $(reductions), $(p)/$(r))), $(foreach a, $(analyses), $(pp)/$(a)/hyperparams.pickle)))

det_metrics_ph := $(foreach ds, $(datasets_list), $(foreach m, $(models), $(foreach r, $(reductions), $(foreach a, $(analyses), det_metrics_ph-$(ds)-$(m)-$(r)-$(a)))))

det_metrics_others := $(foreach ds, $(datasets_list), $(foreach m, $(models), det_metrics_others-$(ds)-$(m)))

# ----------------
#  Rules
# ----------------
.PHONY: xp_det_metrics_ph xp_det_metrics_others $(det_metrics_ph) $(det_metrics_others)
#
# ----------------
# Accessible Rules
# ----------------
xp_datasets: $(datasets)

xp_corevectors: $(corevectors)

xp_tuning: $(tunings)

xp_det_metrics_ph: $(det_metrics_ph)

xp_det_metrics_others: $(det_metrics_others)

# ----------------
# Actual Rules
# ----------------

#  Datasets
define ds_template =
$(1):
	python datasets/xp_datasets.py -ds $(2) -m $(3) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach m, $(models), $(eval $(call ds_template, $(DATA_DIR)/$(ds)/datasets/dss.$(ds)-test.APGD-t-$(m), $(ds), $(m)))))

# Corevectors
define cvs_template =
$(1):
	python corevectors/xp_corevectors.py -m $(2) -r $(3) -ds $(4) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach r, $(reductions), $(foreach m, $(models), $(eval $(call cvs_template, $(DATA_DIR)/$(ds)/$(m)/corevectors/$(r), $(m), $(r), $(ds))))))

# Tunings
define tune_template =
$(1):
	python tuning/xp_tuning.py -m $(2) -r $(3) -a $(4) -ds $(5) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach a, $(analyses), $(foreach r, $(reductions), $(foreach m, $(models), $(eval $(call tune_template, $(DATA_DIR)/$(ds)/$(m)/peepholes/$(r)/$(a)/hyperparams.pickle, $(m), $(r), $(a), $(ds)))))))

# Detection metrics - peepholes
define det_ph_template =
$(1):
	python det_metrics/xp_peepholes.py -m $(3) -r $(4) -a $(5) -ds $(2) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach m, $(models), $(foreach r, $(reductions), $(foreach a, $(analyses), $(eval $(call det_ph_template, det_metrics_ph-$(ds)-$(m)-$(r)-$(a), $(ds), $(m), $(r), $(a)))))))

# Detection metrics - others
define det_others_template =
$(1):
	python det_metrics/xp_others.py -m $(3) -ds $(2) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach m, $(models), $(eval $(call det_others_template, det_metrics_others-$(ds)-$(m), $(ds), $(m)))))
