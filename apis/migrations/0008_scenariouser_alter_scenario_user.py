# Generated by Django 4.1.5 on 2023-01-08 11:30

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('apis', '0007_remove_scenario_description_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='ScenarioUser',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.CharField(blank=True, max_length=100)),
            ],
        ),
        migrations.AlterField(
            model_name='scenario',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='Scenario', to='apis.scenariouser'),
        ),
    ]