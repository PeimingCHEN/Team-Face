from django.shortcuts import render
from accounts.models import User, Organization, InvitationCode


def home_view(request):
    context = {}
    return render(request, "home-view.html", context)
