#!/bin/bash

#### =====================================================================================
#### ==== 00. Configure. Set variables here and where marked with `ACT: ==================
#### =====================================================================================
DIR_DATA_LOCAL=#ACT:SET_THE_VARIABLE
DIR_PROJECT_ROOT=#ACT:SET_THE_VARIABLE
DIR_PROJECT_RESULTS=${DIR_PROJECT_ROOT}/results

#### =====================================================================================
#### ==== 01. Select samples/imaging, define target, and extract from OAI ================
#### =====================================================================================

# Follow `Targets_meta_and_scans_from_OAI.ipynb`
# Targets are in `tiulpin2019multimodal__labels.csv`

#### =====================================================================================
#### ==== 02. Prepare dataset ============================================================
#### =====================================================================================

(echo "prepare_dataset" &&
 python prepare_data_mri_oai.py \
  dir_root_oai_mri=${ACT:SET_DIR_DATA_IN_DESS} \
  dir_root_output=${ACT:SET_DIR_DATA_OUT_DESS} \
  path_csv_extract=${ACT:SET_PATH_META_EXTRACT_DESS} \
  num_threads=24 \
)

(echo "prepare_dataset" &&
 python prepare_data_xr_mipt.py \
   dir_root_mipt_xr=${ACT:SET_DIR_DATA_IN_XR} \
   dir_root_output=${ACT:SET_DIR_DATA_OUT_XR} \
   num_threads=24 \
)

#### =====================================================================================
#### ==== 03. Train models ===============================================================
#### =====================================================================================

# The models were trained on a cluster, with the hyper-parameters (e.g. batch size) set
# considering the hardware - 64 CPU cores, 320GB RAM, 4xNVIDIA A100 GPU, 256GB NVME
# storage). Adjust the batch sizes according to your setup

BATCH_SIZE=16
for FE_PRETRAIN in true false ; do
(\
echo cnn2d_fc fe_pret-${FE_PRETRAIN} && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn2d_fc model.fe.pretrained='"${FE_PRETRAIN}"\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)
done

BATCH_SIZE=32
for FE_PRETRAIN in true false ; do
(\
echo cnn2d_lstm fe_pret-${FE_PRETRAIN} && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn2d_lstm model.fe.pretrained='"${FE_PRETRAIN}"\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)
done

BATCH_SIZE=32
for FE_PRETRAIN in true false ; do
(\
echo cnn2d_trf rc fe_pret-${FE_PRETRAIN} && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn2d_trf model.fe.dims_view=rc model.fe.pretrained='"${FE_PRETRAIN}"\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)
done

BATCH_SIZE=32
(\
echo cnn2d_trf rs && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn2d_trf model.fe.dims_view=rs model.fe.pretrained=true'\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.25,1.0]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)

BATCH_SIZE=32
(\
echo cnn2d_trf cs && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn2d_trf model.fe.dims_view=cs model.fe.pretrained=true'\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.25,0.5,1.0]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)

BATCH_SIZE=8
for FE_PRETRAIN in true false ; do
(\
echo cnn2d_multiview_trf shared=false fe_pret-${FE_PRETRAIN} && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn2d_multiview_trf model.fe.shared=false model.fe.pretrained='"${FE_PRETRAIN}"\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)
done

BATCH_SIZE=8
for FE_PRETRAIN in true false ; do
(\
echo cnn2d_multiview_trf shared=true fe_pret-${FE_PRETRAIN} && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn2d_multiview_trf model.fe.shared=true model.fe.pretrained='"${FE_PRETRAIN}"\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)
done

BATCH_SIZE=32
(\
echo cnn2p1d_fc && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn2p1d_fc'\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)

BATCH_SIZE=64
(\
echo cnn3d_fc && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn3d_fc'\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)

BATCH_SIZE=64
(\
echo shufflenet3d_fc && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=shufflenet3d_fc'\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)

BATCH_SIZE=8
(\
echo cnn3d_fc l1_stride=1 && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=cnn3d_fc'\
' data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]]'\
' model.fe.l1_stride=1 '\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)

BATCH_SIZE=64
(\
echo xr_cnn && \
LABEL_TIME=`date +%y%m%d_%H%M`; LABEL_RAND=`openssl rand -hex 2` && \
python train_prog.py \
' path_project_root='"${DIR_PROJECT_ROOT}"' path_data_root='"${DIR_DATA_LOCAL}"\
' path_experiment_root='"${DIR_PROJECT_RESULTS}/${LABEL_TIME}__${LABEL_RAND}"\
' model=xr_cnn'\
' data.sets.n0.modals=[xr_pa] model.input_size=[[700,700]] model.downscale=[[0.5,0.5]]'\
' training.optim.lr_init=1e-3'\
' +training.sched={name:CustomWarmupMultiStepLR,params:{epochs_warmup:5,mstep_milestones:[20,40]}}'\
' training.batch_size='"${BATCH_SIZE}"' validation.batch_size='"${BATCH_SIZE}"\
)


#### =====================================================================================
#### ==== 04. Evaluate models on hold out data ===========================================
#### =====================================================================================

# 1. Evaluation
PROFILE=none
# 2. MACs, num params
#PROFILE=compute
# 3. Inference time
#PROFILE=time
#BATCH_SIZE=1

#ACT:SELECT_ONE
FE_PRETRAIN=true ; EXPERIM=#ACT:SET_THE_VARIABLE
FE_PRETRAIN=false ; EXPERIM=#ACT:SET_THE_VARIABLE
BATCH_SIZE=8
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_fc model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
FE_PRETRAIN=true ; EXPERIM=#ACT:SET_THE_VARIABLE  
FE_PRETRAIN=false ; EXPERIM=#ACT:SET_THE_VARIABLE  
BATCH_SIZE=16
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_lstm model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
FE_PRETRAIN=true ; EXPERIM=#ACT:SET_THE_VARIABLE 
FE_PRETRAIN=false ; EXPERIM=#ACT:SET_THE_VARIABLE 
BATCH_SIZE=16
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_trf model.fe.dims_view=rc model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
EXPERIM=#ACT:SET_THE_VARIABLE
BATCH_SIZE=16
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_trf model.fe.dims_view=rs model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
EXPERIM=#ACT:SET_THE_VARIABLE
BATCH_SIZE=16
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_trf model.fe.dims_view=rs model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.25,1.0]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
EXPERIM=#ACT:SET_THE_VARIABLE
BATCH_SIZE=16
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_trf model.fe.dims_view=cs model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
EXPERIM=#ACT:SET_THE_VARIABLE
BATCH_SIZE=16
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_trf model.fe.dims_view=cs model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.25,0.5,1.0]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
FE_PRETRAIN=true ; EXPERIM=#ACT:SET_THE_VARIABLE 
FE_PRETRAIN=false ; EXPERIM=#ACT:SET_THE_VARIABLE 
BATCH_SIZE=2
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_multiview_trf model.fe.shared=false model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
FE_PRETRAIN=true ; EXPERIM=#ACT:SET_THE_VARIABLE 
FE_PRETRAIN=false ; EXPERIM=#ACT:SET_THE_VARIABLE 
BATCH_SIZE=2
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2d_multiview_trf model.fe.shared=true model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

EXPERIM=#ACT:SET_THE_VARIABLE
BATCH_SIZE=4
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn2p1d_fc model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#ACT:SELECT_ONE
L1_STRIDE=2 ; EXPERIM=#ACT:SET_THE_VARIABLE 
L1_STRIDE=1 ; EXPERIM=#ACT:SET_THE_VARIABLE 
BATCH_SIZE=8
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=cnn3d_fc model.fe.l1_stride=${L1_STRIDE} model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

EXPERIM=#ACT:SET_THE_VARIABLE
BATCH_SIZE=8
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=shufflenet3d_fc model.restore_weights=true \
    data.sets.n0.modals=[sag_3d_dess] model.input_size=[[320,320,128]] model.downscale=[[0.5,0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

EXPERIM=#ACT:SET_THE_VARIABLE
BATCH_SIZE=32
(\
  echo "eval_prog" ${EXPERIM} &&
  python eval_prog.py \
    path_project_root=${DIR_PROJECT_ROOT} path_data_root=${DIR_DATA_LOCAL} \
    path_experiment_root=${DIR_PROJECT_RESULTS}/${EXPERIM} \
    model=xr_cnn model.restore_weights=true \
    data.sets.n0.modals=[xr_pa] model.input_size=[[700,700]] model.downscale=[[0.5,0.5]] \
    testing.batch_size=${BATCH_SIZE} \
    testing.profile=${PROFILE} \
)

#### =====================================================================================
#### ==== 05. Analyze predictions and compare models =====================================
#### =====================================================================================

# Follow `Analysis_Visualization.ipynb`
