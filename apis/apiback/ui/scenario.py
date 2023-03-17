from rest_framework.views import APIView
from rest_framework.response import Response
from ...models import Scenario


class scenario(APIView):
    def get(self, request, scenarioId):

        print('id:', scenarioId)
        scenario = Scenario.objects.get(id=scenarioId)

        if (scenario is not None):
            return Response({
                'scenarioName': scenario.scenario_name,
                'description': scenario.description,
            }, status=200)
        else:
            return Response("Not Exist", status=201)

    def put(self, request):
        scenario = Scenario.objects.get(
            id=request.data['id'])

        scenario.scenario_name = request.data['name']
        scenario.description = request.data['description']
        scenario.save()

        return Response("successfully changed")
