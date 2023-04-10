from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager


class CustomUserManager(BaseUserManager):
    def create_user(self, username, email, password=None, is_admin=False, **extra_fields):
        """
        Creates and saves a User with the given email, username and password.
        """
        if not username:
            raise ValueError('The Username field must be set')
        if not email:
            raise ValueError('The Email field must be set')

        user = self.model(
            username=username,
            email=self.normalize_email(email),
            is_admin=is_admin,
            **extra_fields
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, username, email, password=None, **extra_fields):
        """
        Creates and saves a superuser with the given email, username and password.
        """
        extra_fields.setdefault('is_admin', True)

        if extra_fields.get('is_admin') is not True:
            raise ValueError('Superuser must have is_admin=True.')

        return self.create_user(username, email, password, **extra_fields)


class CustomUser(AbstractBaseUser):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=50, unique=True)
    email = models.EmailField(max_length=255, unique=True)
    password = models.CharField(max_length=500)
    is_admin = models.BooleanField(default=False)

    USERNAME_FIELD = 'username'
    EMAIL_FIELD = 'email'
    REQUIRED_FIELDS = ['email']

    objects = CustomUserManager()

    def __str__(self):
        return self.username

    def has_perm(self, perm, obj=None):
        return True

    def has_module_perms(self, app_label):
        return True

    def get_user_name(self):
        return self.username

    def get_user_password(self):
        return self.password

    def get_user_id(self):
        return f"{self.id}"

    @property
    def is_staff(self):
        return self.is_admin


class Scenario(models.Model):
    id = models.AutoField(primary_key=True)
    user = models.ForeignKey(CustomUser,
                             on_delete=models.CASCADE)
    scenario_name = models.CharField(max_length=100, null=False)
    description = models.CharField(max_length=200)

    class Meta:
        unique_together = ('user', 'scenario_name',)

    def __str__(self):
        return f"{self.pk}.{self.user}"

    def get_scenario_name(self):
        return self.scenario_name

    def get_description(self):
        return self.description

    def get_user_id(self):
        return self.user_id

    def get_scenario_id(self):
        return f"{id}"


class ScenarioSolution(models.Model):
    id = models.AutoField(primary_key=True)
    SOLUTION_TYPE_CHOISES = (
        ('supervised', 'Supervised'),
        ('unsupervised', 'Unsupervised')
    )
    user = models.ForeignKey(CustomUser,
                             on_delete=models.CASCADE)
    scenario = models.ForeignKey(Scenario, on_delete=models.CASCADE)
    solution_name = models.CharField(max_length=200, unique=True, null=False)
    description = models.TextField(blank=True, null=True)
    solution_type = models.CharField(
        max_length=20, choices=SOLUTION_TYPE_CHOISES, null=False)
    training_file = models.FileField(upload_to='files', blank=True, null=True)
    test_file = models.FileField(upload_to='files', blank=True, null=True)
    protected_features = models.CharField(
        max_length=200, blank=True, null=True)
    protected_values = models.CharField(max_length=200, blank=True, null=True)
    target_column = models.CharField(
        max_length=200, null=True, default='target')
    outlier_data_file = models.FileField(
        upload_to='files', blank=True, null=True)
    favourable_outcome = models.CharField(
        max_length=200, blank=True, default='1')
    factsheet_file = models.FileField(upload_to='files', blank=True, null=True)
    model_file = models.FileField(upload_to='files', blank=True, null=True)
    metrics_mappings_file = models.FileField(
        upload_to='files', blank=True, null=True)
    weights_metrics = models.FileField(
        upload_to='files', blank=True, null=True)
    weights_pillars = models.FileField(
        upload_to='files', blank=True, null=True)

    class Meta:
        unique_together = ('user', 'scenario', 'solution_name')

    def __str__(self):
        return f"{self.pk}.{self.solution_name}"

    def get_user_id(self):
        return f"{self.user_id}"

    def get_scenario_id(self):
        return self.scenario_id

    def get_solution_name(self):
        return self.solution_name

    def get_description(self):
        return self.description

    def get_solution_type(self):
        return self.solution_type

    def get_traing_file(self):
        return self.training_file

    def get_test_file(self):
        return self.test_file

    def get_protected_value(self):
        return self.protected_values

    def get_protected_feature(self):
        return self.protected_features

    def get_target_column(self):
        return self.target_column

    def get_outlier_file(self):
        return self.outlier_data_file

    def get_favourable_outcome(self):
        return self.favourable_outcome

    def get_factsheet_file(self):
        return self.factsheet_file

    def get_model_file(self):
        return self.model_file

    def get_metrics_mapping_file(self):
        return self.metrics_mappings_file

    def get_weights_metrics(self):
        return self.weights_metrics

    def get_weights_pillars(self):
        return self.weights_pillars


class Score(models.Model):
    id = models.AutoField(primary_key=True)
    solution = models.ForeignKey(ScenarioSolution, on_delete=models.CASCADE)

    # Fairness
    fairness = models.FloatField(null=True)
    underfitting = models.FloatField(null=True)
    overfitting = models.FloatField(null=True)
    statistical = models.FloatField(null=True)
    disparate = models.FloatField(null=True)
    equal = models.FloatField(null=True)
    average = models.FloatField(null=True)
    class_balance = models.FloatField(null=True)

    # explainability
    explain = models.FloatField(null=True)
    correlated = models.FloatField(null=True)
    model_size = models.FloatField(null=True)
    permutation = models.FloatField(null=True)
    feature = models.FloatField(null=True)
    algorithm = models.FloatField(null=True)

    # robustness
    robust = models.FloatField(null=True)
    clever = models.FloatField(null=True)
    confidence = models.FloatField(null=True)
    clique = models.FloatField(null=True)
    er_fast = models.FloatField(null=True)
    er_carlini = models.FloatField(null=True)
    er_deep = models.FloatField(null=True)
    loss_sensitivity = models.FloatField(null=True)

    # accountability
    account = models.FloatField(null=True)
    train_test_split = models.FloatField(null=True)
    missing_data = models.FloatField(null=True)
    normalization = models.FloatField(null=True)
    regularization = models.FloatField(null=True)
    factsheet = models.FloatField(null=True)

    trust = models.FloatField(null=True)


class MetricProperty(models.Model):
    id = models.AutoField(primary_key=True)
    solution = models.ForeignKey(ScenarioSolution, on_delete=models.CASCADE)

    underfitting_property = models.JSONField(max_length=1000, null=True)
    overfitting_property = models.JSONField(max_length=1000, null=True)
    statistical_parity_difference_property = models.JSONField(
        max_length=1000, null=True)
    equal_opportunity_difference_property = models.JSONField(
        max_length=1000, null=True)
    average_odds_difference_property = models.JSONField(
        max_length=1000, null=True)
    disparate_impact_property = models.JSONField(max_length=1000, null=True)
    class_balance_property = models.JSONField(max_length=1000, null=True)
    algorithm_class_property = models.JSONField(max_length=1000, null=True)
    correlated_features_property = models.JSONField(max_length=1000, null=True)
    model_size_property = models.JSONField(max_length=1000, null=True)
    feature_relevance_property = models.JSONField(max_length=1000, null=True)
    permutation_feature_importance_property = models.JSONField(
        max_length=1000, null=True)
    normalization_property = models.JSONField(max_length=1000, null=True)
    missing_data_property = models.JSONField(max_length=1000, null=True)
    regularization_property = models.JSONField(max_length=1000, null=True)
    train_test_split_property = models.JSONField(max_length=1000, null=True)
    factsheet_completeness_property = models.JSONField(
        max_length=1000, null=True)
    confidence_score_property = models.JSONField(max_length=1000, null=True)
    clique_method_property = models.JSONField(max_length=1000, null=True)
    clever_score_property = models.JSONField(max_length=1000, null=True)
    er_fast_gradient_attack_property = models.JSONField(
        max_length=1000, null=True)
    er_carlini_wagner_attack_property = models.JSONField(
        max_length=1000, null=True)
    er_deepfool_attack_property = models.JSONField(max_length=1000, null=True)
    loss_sensitivity_property = models.JSONField(max_length=1000, null=True)


class PerformanceMetrics(models.Model):
    id = models.AutoField(primary_key=True)
    solution = models.ForeignKey(ScenarioSolution, on_delete=models.CASCADE)

    accuracy = models.FloatField(null=True)
    globalrecall = models.FloatField(null=True)
    classweightedrecall = models.FloatField(null=True)
    globalprecision = models.FloatField(null=True)
    classweightedprecision = models.FloatField(null=True)
    globalf1score = models.FloatField(null=True)
    classweightedf1score = models.FloatField(null=True)
