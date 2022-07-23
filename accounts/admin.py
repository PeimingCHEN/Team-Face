from django.contrib import admin

from .models import User, Organization, InvitationCode

class InvitationCodeLine(admin.StackedInline):
    model = InvitationCode
    extra = 0


# Register your models here.
class UserAdmin(admin.ModelAdmin):
    fieldname = 'User'

    list_display = ['name','organization', 'register_time']
    search_fields = ['name','organization']


class OrganizationAdmin(admin.ModelAdmin):
    fieldname = 'Organization'

    inlines = [InvitationCodeLine]

    list_display = ['name','organization_code', 'created_time']
    search_fields = ['name']



admin.site.register(User, UserAdmin)
admin.site.register(Organization, OrganizationAdmin)


