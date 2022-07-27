from rest_framework.serializers import ModelSerializer
from .models import Organization, User, UserFaceImage, UserTestImage


class OrganizationListSerializer(ModelSerializer):
    class Meta:
        model = Organization
        fields = '__all__'


class MemberSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = ['name', 'email', 'phone', 'register_time']


class OrganizationSerializer(ModelSerializer):
    members = MemberSerializer(many=True, read_only=True)
    class Meta:
        model = Organization
        fields = '__all__'


class UserListSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'


class UserFaceImageSerializer(ModelSerializer):
    class Meta:
        model = UserFaceImage
        fields = '__all__'


class UserTestImageSerializer(ModelSerializer):
    class Meta:
        model = UserTestImage
        fields = '__all__'


class UserSerializer(ModelSerializer):
    images = UserFaceImageSerializer(many=True, read_only=True)
    class Meta:
        model = User
        fields = '__all__'
        extra_kwargs = {
            'organization': {'required': False},
            'name': {'required': False},
            'email': {'required': False},
            'password': {'required': False},
            'phone': {'required': False},
            # 'images':{'required': False}
            }

