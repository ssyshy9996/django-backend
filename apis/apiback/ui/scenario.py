from rest_framework.views import APIView
from rest_framework.response import Response
from ...models import Scenario, CustomUser


class scenario(APIView):
    def get(self, request, scenarioId):
        scenario = Scenario.objects.get(id=scenarioId)

        print('id:', scenario.description, scenario.scenario_name)
        if (scenario is not None):
            return Response({
                'scenarioName': scenario.scenario_name,
                'description': scenario.description,
            }, status=200)
        else:
            return Response("Not Exist", status=201)

    def delete(self, request, scenarioId):
        print('id:', scenarioId)
        Scenario.objects.get(id=scenarioId).delete()
        return Response('Delete Success', status=200)

    def put(self, request):
        scenario = Scenario.objects.get(
            id=request.data['id'])

        scenario.scenario_name = request.data['name']
        scenario.description = request.data['description']
        scenario.save()

        return Response("successfully changed")

    def post(self, request):
        # print('data:', request.data['ScenarioName'])
        # scenario = Scenario.objects.get(
        #     scenario_name=request.data['ScenarioName']
        # )

        # if (scenario is not None):
        #     return Response({
        #         'errorText': 'Same Scenario Name exists, Please try again with different name',
        #     }, status=204)

        user = CustomUser.objects.get(email=request.data['emailid'])
        # isExist = Scenario.objects.get(
        #     scenario_name=request.data['ScenarioName'],
        #     use_id=user.id
        # )

        # if isExist:
        #     return Response({'Save Failed'}, status=400)

        try:
            newScenario = Scenario.objects.create(
                scenario_name=request.data['ScenarioName'],
                description=request.data['Description'],
                user_id=user.id,
            )

            newScenario.save()

            return Response({'Save Success'}, status=200)
        except Exception as e:
            print('except:', e)
            return Response({'Save Failed'}, status=400)


class scenario_list(APIView):
    def get(self, request, email):
        print('email:', email)
        user = CustomUser.objects.get(email=email)
        scenarios = Scenario.objects.filter(user_id=user.id).values()
        uploaddic = {}
        uploaddic['scenarioList'] = scenarios
        return Response(uploaddic, status=200)
