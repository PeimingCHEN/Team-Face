from rest_framework.serializers import ModelSerializer
from .models import Organization, User, UserFaceImage, InvitationCode


class OrganizationSerializer(ModelSerializer):
    class Meta:
        model = Organization
        field = '__all__'

class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        field = '__all__'

class UserFaceImageSerializer(ModelSerializer):
    class Meta:
        model = UserFaceImage
        field = '__all__'

class InvitationCodeSerializer(ModelSerializer):
    class Meta:
        model = InvitationCode
        field = '__all__'


