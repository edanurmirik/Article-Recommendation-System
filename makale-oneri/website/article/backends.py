from django.contrib.auth.backends import BaseBackend
from .models import UserInformation

class UserInformationBackend(BaseBackend):
    def authenticate(self, request, email=None, password=None):
        try:
            user = UserInformation.objects.get(email=email, password=password)
            return user
        except UserInformation.DoesNotExist:
            return None
