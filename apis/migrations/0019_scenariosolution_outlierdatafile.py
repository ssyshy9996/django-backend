# Generated by Django 4.1.5 on 2023-02-19 10:53

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('apis', '0018_scenariosolution_favoutcome_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='scenariosolution',
            name='Outlierdatafile',
            field=models.FileField(blank=True, null=True, upload_to='files'),
        ),
    ]
