# Generated by Django 4.1.5 on 2023-03-28 05:39

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('apis', '0005_score_account_score_explain_score_fariness_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='score',
            old_name='fariness',
            new_name='fairness',
        ),
    ]
