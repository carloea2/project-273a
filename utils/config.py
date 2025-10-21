from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union

class DataColumnsConfig(BaseModel):
    numeric: List[str]
    categorical_low_card: List[str]
    icd_cols: List[str]
    drug_cols: List[str]
    hospital_col: Optional[str] = None
    specialty_col: Optional[str] = None
    admission_type_col: Optional[str] = None
    admission_source_col: Optional[str] = None
    discharge_disposition_col: Optional[str] = None

class DataFiltersConfig(BaseModel):
    min_los: Optional[int] = None
    max_los: Optional[int] = None
    exclude_discharge_to_ids: List[int] = []
    first_encounter_per_patient: bool = True

class IdentifierColsConfig(BaseModel):
    encounter_id: str
    patient_id: str

class TargetConfig(BaseModel):
    name: str
    positive_values: List[str]

class PreprocessingConfig(BaseModel):
    numeric_imputer: str = "mean"
    categorical_imputer: str = "most_frequent"
    scaler: str = "standard"
    categorical_handling: str = "onehot"  # or "embedding"
    use_unknown_category: bool = True
    unknown_label: str = "UNKNOWN"
    min_freq_for_category: int = 1
    truncate_icd_to_3_digits: bool = True
    map_icd_to_group: bool = True
    map_drug_to_class: bool = True

class SplitsConfig(BaseModel):
    strategy: str = "group_k_fold"
    group_by: str = "patient"
    n_splits: int = 5
    seed: int = 42
    stratify_by_target: bool = True

class DataConfig(BaseModel):
    csv_path: str
    id_mapping_path: Optional[str] = None
    identifier_cols: IdentifierColsConfig
    target: TargetConfig
    filters: DataFiltersConfig = DataFiltersConfig()
    columns: DataColumnsConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    splits: SplitsConfig = SplitsConfig()

class EdgeFeatConfig(BaseModel):
    relation_subtypes_by_status: bool = False
    edge_attr_status: bool = False

class GraphConfig(BaseModel):
    node_types_enabled: Dict[str, bool]
    edge_types_enabled: Dict[str, bool]
    edge_featureing: Dict[str, EdgeFeatConfig] = Field(default_factory=dict)
    feature_dims: Dict[str, Union[int, Dict[str, int]]] = Field(default_factory=dict)
    oov_nodes: bool = True
    artifacts_dir: str = "./artifacts"

class LossConfig(BaseModel):
    type: str = "bce_with_logits"
    pos_weight: Union[str, float] = 1.0

class ModelConfig(BaseModel):
    arch: str
    hidden_dim: int
    num_layers: int
    dropout: float = 0.0
    heads: int = 2
    rgcn_bases: int = 0
    act: str = "relu"
    norm: Optional[str] = None
    use_edge_attr_for_drug_status: bool = False
    loss: LossConfig = LossConfig()

class OptimizerConfig(BaseModel):
    name: str = "Adam"
    lr: float = 0.001
    weight_decay: float = 0.0

class SchedulerConfig(BaseModel):
    name: Optional[str] = None
    warmup_epochs: int = 0

class BatchingConfig(BaseModel):
    batch_size_encounters: int = 128
    fanouts_per_layer_by_relation: Dict[str, List[int]] = Field(default_factory=dict)

class TrainConfig(BaseModel):
    epochs: int
    early_stopping_patience: int
    early_stopping_metric: str = "auprc"
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig = SchedulerConfig()
    batching: BatchingConfig = BatchingConfig()
    val_every: int = 1
    gradient_clip_norm: Optional[float] = None
    deterministic: bool = True
    seed: int = 42
    tensorboard_logdir: str = "./tb_logs"
    save_best_on: str = "auprc"

class ThresholdTuningConfig(BaseModel):
    optimize_for: str = "f1_pos"
    grid: List[float] = Field(default_factory=list)
    calibration: Optional[str] = None

class PlotsConfig(BaseModel):
    roc: bool = True
    pr: bool = True
    calibration: bool = True
    confusion: bool = True
    decision_curves: bool = True

class EvaluationConfig(BaseModel):
    metrics_primary: List[str]
    metrics_secondary: List[str]
    threshold_tuning: ThresholdTuningConfig = ThresholdTuningConfig()
    plots: PlotsConfig = PlotsConfig()
    subgroup_metrics: List[str] = Field(default_factory=list)

class InferenceConfig(BaseModel):
    inductive: bool = True
    build_star_subgraph_on_the_fly: bool = True
    oov_handling: str = "UNKNOWN"
    batch_predict_csv_path: Optional[str] = None
    output_predictions_path: str = "./predictions.csv"

class TabularMLPConfig(BaseModel):
    enabled: bool = False
    hidden_dims: List[int] = Field(default_factory=list)
    dropout: float = 0.0
    epochs: int = 10
    batch_size: int = 256
    optimizer: OptimizerConfig = OptimizerConfig()

class BaselineConfig(BaseModel):
    tabular_mlp: TabularMLPConfig = TabularMLPConfig()
    xgboost: Dict[str, bool] = Field(default_factory=dict)

class Config(BaseModel):
    data: DataConfig
    graph: GraphConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    inference: InferenceConfig
    baseline: BaselineConfig

    @validator("model")
    def check_model_arch(cls, v):
        assert v.arch in {"HGT", "RGCN", "GraphSAGE"}, "Unknown model arch"
        return v
