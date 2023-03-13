from rest_framework.authentication import BaseAuthentication
from django.contrib.auth import get_user_model
from .models import CustomUser


class CustomUserAuthentication(BaseAuthentication):
    def authenticate(self, request):
        import base64

        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if auth_header:
            _, auth = auth_header.split()
            decoded_auth = base64.b64decode(auth).decode('utf-8')
            username, password = decoded_auth.split(':')

            print("AUTHENTICATION USERNAME: ", username)
            print("AUTHENTICATION PASSWORD: ", password)

            if not username or not password:
                return None

            # User = get_user_model()
            try:
                user = CustomUser.objects.get(email=username)
                print('u:', user)
            except CustomUser.DoesNotExist:
                print('user none:')
                return None

            return (user, None)
