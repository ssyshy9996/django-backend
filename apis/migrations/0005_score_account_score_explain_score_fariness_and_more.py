# Generated by Django 4.1.5 on 2023-03-28 04:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apis', '0004_score'),
    ]

    operations = [
        migrations.AddField(
            model_name='score',
            name='account',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='score',
            name='explain',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='score',
            name='fariness',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='score',
            name='robust',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='score',
            name='trust',
            field=models.FloatField(null=True),
        ),
    ]
