# Generated by Django 4.1.5 on 2023-04-10 17:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apis', '0007_performancemetrics_metricproperty'),
    ]

    operations = [
        migrations.AlterField(
            model_name='metricproperty',
            name='algorithm_class_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='average_odds_difference_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='class_balance_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='clever_score_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='clique_method_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='confidence_score_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='correlated_features_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='disparate_impact_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='equal_opportunity_difference_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='er_carlini_wagner_attack_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='er_deepfool_attack_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='er_fast_gradient_attack_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='factsheet_completeness_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='feature_relevance_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='loss_sensitivity_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='missing_data_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='model_size_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='normalization_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='overfitting_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='permutation_feature_importance_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='regularization_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='statistical_parity_difference_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='train_test_split_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
        migrations.AlterField(
            model_name='metricproperty',
            name='underfitting_property',
            field=models.JSONField(max_length=1000, null=True),
        ),
    ]
