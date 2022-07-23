from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import (
    OrganizationSerializer,
    UserSerializer,
    UserFaceImageSerializer,
    InvitationCodeSerializer
)
from .models import (
    Organization,
    User,
    UserFaceImage,
    InvitationCode
)
from rest_framework.parsers import JSONParser
from rest_framework import status
from rest_framework.views import APIView
from django.http import Http404


# Create your views here.
@api_view(['GET','POST'])
def get_routes(request):
    routes = {}
    return Response(routes)


class organization_list_apiview(APIView):

    def get(self, request, format=None):
        organizations = Organization.objects.all()
        serializer = OrganizationSerializer(organizations, many=True)
        return Response(serializer.data, safe=False)

    def post(self, request, format=None):
        serializer = OrganizationSerializer(data=JSONParser().parse(request))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                                status=status.HTTP_201_CREATED)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)

    
class organization_apiview(APIView):
    def get_object(self, name):
        try:
            return Organization.objects.get(name=name)
        except Organization.DoesNotExist:
            raise Http404

    def get(self, request, name, format=None):
        organization = self.get_object(name)
        serializer = OrganizationSerializer(organization)
        return Response(serializer.data)

    def put(self, request, name, format=None):
        organization = self.get_object(name)
        serializer = OrganizationSerializer(organization,
                                        data=JSONParser().parse(request))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, name, format=None):
        organization = self.get_object(name)
        organization.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class user_list_apiview(APIView):

    def get(self, request, format=None):
        users = User.objects.all()
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data, safe=False)

    def post(self, request, format=None):
        serializer = UserSerializer(data=JSONParser().parse(request))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                                status=status.HTTP_201_CREATED)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)

class user_apiview(APIView):
    def get_object(self, name):
        try:
            return User.objects.get(name=name)
        except User.DoesNotExist:
            raise Http404

    def get(self, request, name, format=None):
        user = self.get_object(name)
        serializer = UserSerializer(user)
        return Response(serializer.data)

    def put(self, request, name, format=None):
        user = self.get_object(name)
        serializer = UserSerializer(user,
                                        data=JSONParser().parse(request))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, name, format=None):
        user = self.get_object(name)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


class invitationcode_list_apiview(APIView):

    def get(self, request, format=None):
        invitations = InvitationCode.objects.all()
        serializer = InvitationCodeSerializer(invitations, many=True)
        return Response(serializer.data, safe=False)

    def post(self, request, format=None):
        serializer = InvitationCodeSerializer(data=JSONParser().parse(request))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                                status=status.HTTP_201_CREATED)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)

class invitationcode_apiview(APIView):
    def get_object(self, code):
        try:
            return InvitationCode.objects.get(code=code)
        except InvitationCode.DoesNotExist:
            raise Http404

    def get(self, request, code, format=None):
        invitation_code = self.get_object(code)
        serializer = InvitationCodeSerializer(invitation_code)
        return Response(serializer.data)

    def put(self, request, code, format=None):
        invitation_code = self.get_object(code)
        serializer = InvitationCodeSerializer(invitation_code,
                                        data=JSONParser().parse(request))
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, code, format=None):
        invitation_code = self.get_object(code)
        invitation_code.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

