from django.contrib import admin
from accounts.models import User, Organization, UserFaceImage

# class InvitationCodeLine(admin.StackedInline):
#     model = InvitationCode
#     extra = 0


# Register your models here.
class UserFaceImageInline(admin.TabularInline):
    model = UserFaceImage
    extra = 0


class UserAdmin(admin.ModelAdmin):
    fieldname = 'User'
    inlines = (UserFaceImageInline,)
    list_display = ['name', 'organization', 'register_time']
    search_fields = ['name','organization']


class UserInline(admin.TabularInline):
    model = User
    extra = 0


class OrganizationAdmin(admin.ModelAdmin):
    fieldname = 'Organization'
    inlines = (UserInline,)
    list_display = ['name', 'organization_code', 'created_time']
    search_fields = ['name']


admin.site.register(User, UserAdmin)
admin.site.register(Organization, OrganizationAdmin)


