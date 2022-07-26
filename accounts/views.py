from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import Http404
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import permission_classes
from accounts.serializers import (
    OrganizationListSerializer,
    OrganizationSerializer,
    UserListSerializer,
    UserSerializer,
)
from accounts.models import (
    Organization,
    User,
)


# @permission_classes([IsAuthenticated])
class organization_list_apiview(APIView):
    def get(self, request, format=None):
        organizations = Organization.objects.all()
        serializer = OrganizationListSerializer(organizations, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        serializer = OrganizationListSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                                status=status.HTTP_201_CREATED)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)


# @permission_classes([IsAuthenticated])
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
                                        data=request.data)
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
        serializer = UserListSerializer(users, many=True)
        return Response(serializer.data)

    def post(self, request, format=None):
        # 用户注册功能
        # 获得填写的邀请码
        invitation_code = request.data.get('organization')
        try:
            # 根据邀请码获取组织
            organization = Organization.objects.get(invitation_code=invitation_code)
        except Organization.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        data=request.data.copy()
        data['organization'] = organization.name
        serializer = UserListSerializer(data=data)
        # 注册用户
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data,
                                status=status.HTTP_201_CREATED)
        return Response(serializer.errors,
                            status=status.HTTP_400_BAD_REQUEST)


# @permission_classes([IsAuthenticated])
class user_apiview(APIView):
    def get_object(self, phone):
        try:
            return User.objects.get(phone=phone)
        except User.DoesNotExist:
            raise Http404

    def get(self, request, phone, format=None):
        user = self.get_object(phone)
        serializer = UserSerializer(user)
        return Response(serializer.data)

    def post(self, request, phone, format=None):
        # 用户登录功能
        user = self.get_object(phone)
        serializer = UserSerializer(user)
        # 获得response中密码
        password = request.data.get('password')
        if user.password == password:
            # 匹配成功则登录成功
            return Response(serializer.data)
        else:
            return Response(status=status.HTTP_404_NOT_FOUND)

    def put(self, request, phone, format=None):
        user = self.get_object(phone)
        serializer = UserSerializer(user,
                                    data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors,
                        status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, phone, format=None):
        user = self.get_object(phone)
        user.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# class invitationcode_list_apiview(APIView):

#     def get(self, request, format=None):
#         invitations = InvitationCode.objects.all()
#         serializer = InvitationCodeSerializer(invitations, many=True)
#         return Response(serializer.data, safe=False)

#     def post(self, request, format=None):
#         serializer = InvitationCodeSerializer(data=JSONParser().parse(request))
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data,
#                                 status=status.HTTP_201_CREATED)
#         return Response(serializer.errors,
#                             status=status.HTTP_400_BAD_REQUEST)

# class invitationcode_apiview(APIView):
#     def get_object(self, code):
#         try:
#             return InvitationCode.objects.get(code=code)
#         except InvitationCode.DoesNotExist:
#             raise Http404

#     def get(self, request, code, format=None):
#         invitation_code = self.get_object(code)
#         serializer = InvitationCodeSerializer(invitation_code)
#         return Response(serializer.data)

#     def put(self, request, code, format=None):
#         invitation_code = self.get_object(code)
#         serializer = InvitationCodeSerializer(invitation_code,
#                                         data=JSONParser().parse(request))
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data)
#         return Response(serializer.errors,
#                             status=status.HTTP_400_BAD_REQUEST)

#     def delete(self, request, code, format=None):
#         invitation_code = self.get_object(code)
#         invitation_code.delete()
#         return Response(status=status.HTTP_204_NO_CONTENT)

