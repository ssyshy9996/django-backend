from rest_framework.authentication import BaseAuthentication
from django.contrib.auth import get_user_model
from .models import CustomUser

from django.contrib.auth.hashers import make_password
from passlib.handlers.django import django_pbkdf2_sha256

password = 'testpassword123'

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
            print('password-encoded:', password)

            if not username or not password:
                return None

            # User = get_user_model()
            # here.. should be inserted.
            try:
                user = CustomUser.objects.get(email=username)
                
                print('u:', user.password)
                # django_hash = make_password(password)   
                is_verified = django_pbkdf2_sha256.verify(password, user.password)
                print('value:', is_verified)
                if is_verified:
                    print('user:', user)
                    return (user, None)
                else:
                    return None
            except CustomUser.DoesNotExist:
                print('user none:')
                return None
